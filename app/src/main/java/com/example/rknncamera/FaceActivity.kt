package com.example.rknncamera

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.TextureView
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import java.util.concurrent.Executors

class FaceActivity : Activity() {
    private lateinit var nameInput: EditText
    private lateinit var serialNumberInput: EditText
    private lateinit var activateButton: Button
    private lateinit var status: TextView
    private lateinit var enrollButton: Button
    private lateinit var identifyButton: Button
    private lateinit var cameraEnrollButton: Button
    private lateinit var cameraIdentifyButton: Button
    private lateinit var cameraPreviewView: TextureView
    private val engine = FaceSdkEngine()
    private lateinit var store: FaceFeatureStore
    private val worker = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var cameraPreview: FaceCameraPreview
    private var liveMode: LiveMode = LiveMode.Idle
    private var activationTimeout: Runnable? = null
    private var activationAttempt = 0
    private var sdkReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_face)
        store = FaceFeatureStore(this)
        serialNumberInput = findViewById(R.id.face_serial_number)
        activateButton = findViewById(R.id.face_activate)
        nameInput = findViewById(R.id.face_name)
        status = findViewById(R.id.face_status)
        enrollButton = findViewById(R.id.face_enroll)
        identifyButton = findViewById(R.id.face_identify)
        cameraPreviewView = findViewById(R.id.face_camera_preview)
        cameraEnrollButton = findViewById(R.id.face_camera_enroll)
        cameraIdentifyButton = findViewById(R.id.face_camera_identify)
        cameraPreview = FaceCameraPreview(this, cameraPreviewView, worker, ::handleLiveFrame, ::renderStatus)
        activateButton.setOnClickListener { activateAndInitialize() }
        enrollButton.setOnClickListener { selectImage(REQUEST_ENROLL) }
        identifyButton.setOnClickListener { selectImage(REQUEST_IDENTIFY) }
        cameraEnrollButton.setOnClickListener(::startCameraEnroll)
        cameraIdentifyButton.setOnClickListener(::startCameraIdentify)
        findViewById<Button>(R.id.face_clear).setOnClickListener {
            store.clear()
            renderStatus("已清空测试人脸库")
        }
        setActionEnabled(false)
        serialNumberInput.setText(DEFAULT_SERIAL_NUMBER)
        cameraPreview.start()
        renderStatus("请确认序列号后，点击“激活并加载 SDK”")
    }

    override fun onDestroy() {
        activationTimeout?.let(mainHandler::removeCallbacks)
        cameraPreview.close()
        worker.shutdownNow()
        super.onDestroy()
    }

    @Deprecated("Uses the Activity result API available in this legacy Activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        val image = data?.data?.let(::decodeBitmap)
        if (image == null) {
            renderStatus("图片读取失败")
            return
        }
        when (requestCode) {
            REQUEST_ENROLL -> enroll(image)
            REQUEST_IDENTIFY -> identify(image)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == FaceCameraPreview.REQUEST_CAMERA && grantResults.firstOrNull() == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            cameraPreview.start()
        }
    }

    private fun activateAndInitialize() {
        val serialNumber = serialNumberInput.text.toString().trim().uppercase()
        if (!SERIAL_NUMBER_PATTERN.matches(serialNumber)) {
            renderStatus("请输入格式正确的序列号，例如 XXXX-XXXX-XXXX-XXXX")
            return
        }
        val attempt = ++activationAttempt
        sdkReady = false
        activationTimeout?.let(mainHandler::removeCallbacks)
        setActionEnabled(false)
        activateButton.isEnabled = false
        val handleState: (FaceSdkEngine.State) -> Unit = { state ->
            runOnUiThread {
                if (attempt != activationAttempt) return@runOnUiThread
                when (state) {
                    FaceSdkEngine.State.Activating -> renderStatus("正在提交序列号激活请求…")
                    FaceSdkEngine.State.Loading -> {
                        activationTimeout?.let(mainHandler::removeCallbacks)
                        renderStatus("激活成功，正在加载人脸模型…")
                    }
                    FaceSdkEngine.State.Ready -> {
                        activationTimeout?.let(mainHandler::removeCallbacks)
                        sdkReady = true
                        activateButton.isEnabled = true
                        setActionEnabled(true)
                        renderStatus("激活成功，SDK 就绪，当前人脸库 ${store.count()} 人")
                    }
                    is FaceSdkEngine.State.Failed -> {
                        activationTimeout?.let(mainHandler::removeCallbacks)
                        activateButton.isEnabled = true
                        renderStatus("激活失败（${state.code}）：${activationHint(state)}")
                    }
                }
            }
        }
        activationTimeout = Runnable {
            if (attempt == activationAttempt && !sdkReady) {
                activateButton.isEnabled = true
                renderStatus("激活超时：未收到 SDK 回调，请检查设备网络、序列号授权状态后重试")
            }
        }.also { mainHandler.postDelayed(it, ACTIVATION_CALLBACK_TIMEOUT_MS) }
        engine.activateAndInitialize(this, serialNumber, handleState)
    }

    private fun startCameraEnroll(view: android.view.View) {
        if (liveMode is LiveMode.Enroll) {
            liveMode = LiveMode.Idle
            setActionEnabled(sdkReady)
            cameraEnrollButton.text = "从摄像头录入"
            renderStatus("已停止从摄像头录入")
            return
        }

        val name = nameInput.text.toString().trim()
        if (!sdkReady) return renderStatus("SDK 尚未就绪")
        if (name.isBlank()) return renderStatus("请先填写姓名或测试标识")
        liveMode = LiveMode.Enroll(name)
        setActionEnabled(false)
        cameraEnrollButton.isEnabled = true
        cameraEnrollButton.text = "停止从摄像头录入"
        renderStatus("正在实时检测可录入的人脸，请正对前置摄像头…")
        cameraPreview.requestFrame()
    }

    private fun startCameraIdentify(view: android.view.View) {
        if (liveMode == LiveMode.Identify) {
            liveMode = LiveMode.Idle
            setActionEnabled(sdkReady)
            cameraIdentifyButton.text = "开始实时识别"
            renderStatus("已停止实时识别")
            return
        }
        if (!sdkReady) return renderStatus("SDK 尚未就绪")
        if (store.count() == 0) return renderStatus("人脸库为空，请先录入人脸")
        liveMode = LiveMode.Identify
        cameraIdentifyButton.text = "停止实时识别"
        setActionEnabled(false)
        cameraIdentifyButton.isEnabled = true
        renderStatus("实时识别中，请正对前置摄像头…")
        cameraPreview.requestFrame()
    }

    private fun handleLiveFrame(bitmap: Bitmap) {
        val mode = liveMode
        if (mode == LiveMode.Idle || !sdkReady) {
            bitmap.recycle()
            return
        }
        val result = when (val feature = engine.extract(bitmap)) {
            is FaceSdkEngine.FeatureResult.Failed -> when (mode) {
                is LiveMode.Enroll -> "未检测到可录入的人脸，请保持正脸并靠近摄像头…"
                LiveMode.Identify -> feature.message
                LiveMode.Idle -> ""
            }
            is FaceSdkEngine.FeatureResult.Success -> when (mode) {
                is LiveMode.Enroll -> {
                    if (liveMode != mode) {
                        ""
                    } else {
                        store.save(mode.name, feature.bytes)
                        liveMode = LiveMode.Idle
                        "实时录入成功：${mode.name}；当前人脸库 ${store.count()} 人"
                    }
                }
                LiveMode.Identify -> identifyFeature(feature.bytes)
                LiveMode.Idle -> ""
            }
        }
        bitmap.recycle()
        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread
            if (result.isNotEmpty()) renderStatus(result)
            when (liveMode) {
                is LiveMode.Enroll, LiveMode.Identify -> if (sdkReady) {
                    mainHandler.postDelayed({ cameraPreview.requestFrame() }, LIVE_IDENTIFY_INTERVAL_MS)
                }
                LiveMode.Idle -> {
                    cameraEnrollButton.text = "从摄像头录入"
                    setActionEnabled(sdkReady)
                }
            }
        }
    }

    private fun identifyFeature(feature: ByteArray): String {
        val match = store.bestMatch(feature, engine::similarity)
            ?: return "人脸库为空，请先录入人脸"
        return if (match.score >= MATCH_THRESHOLD) {
            "实时识别：${match.name}，相似度 ${"%.2f".format(match.score)}"
        } else {
            "未识别，最高候选 ${match.name}，相似度 ${"%.2f".format(match.score)}"
        }
    }

    private fun enroll(bitmap: Bitmap) = process(bitmap, "正在录入人脸…") { feature ->
        val name = nameInput.text.toString().trim()
        if (name.isBlank()) return@process "请先填写姓名或测试标识"
        store.save(name, feature)
        "录入成功：$name；当前人脸库 ${store.count()} 人"
    }

    private fun identify(bitmap: Bitmap) = process(bitmap, "正在识别…") { feature ->
        val match = store.bestMatch(feature, engine::similarity)
            ?: return@process "人脸库为空，请先录入人脸"
        if (match.score >= MATCH_THRESHOLD) {
            "识别成功：${match.name}，相似度 ${"%.2f".format(match.score)}"
        } else {
            "未识别到匹配人脸，最高候选 ${match.name}，相似度 ${"%.2f".format(match.score)}"
        }
    }

    private fun process(bitmap: Bitmap, processingText: String, onFeature: (ByteArray) -> String) {
        if (!sdkReady) {
            renderStatus("SDK 尚未就绪")
            bitmap.recycle()
            return
        }
        setActionEnabled(false)
        renderStatus(processingText)
        worker.execute {
            val result = when (val feature = engine.extract(bitmap)) {
                is FaceSdkEngine.FeatureResult.Success -> onFeature(feature.bytes)
                is FaceSdkEngine.FeatureResult.Failed -> feature.message
            }
            bitmap.recycle()
            runOnUiThread {
                setActionEnabled(sdkReady)
                renderStatus(result)
            }
        }
    }

    private fun selectImage(requestCode: Int) {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "image/*"
            },
            requestCode
        )
    }

    private fun decodeBitmap(uri: android.net.Uri): Bitmap? = runCatching {
        contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
    }.getOrNull()

    private fun setActionEnabled(enabled: Boolean) {
        enrollButton.isEnabled = enabled
        identifyButton.isEnabled = enabled
        cameraEnrollButton.isEnabled = enabled
        cameraIdentifyButton.isEnabled = enabled
    }

    private fun renderStatus(text: String) {
        status.text = text
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show()
    }

    private fun activationHint(state: FaceSdkEngine.State.Failed): String = when {
        state.message.equals("key invalid", ignoreCase = true) -> "序列号无效"
        state.message.equals("license has actived on other device", ignoreCase = true) -> "该序列号已在其他设备激活"
        state.message.equals("auth expired time", ignoreCase = true) || state.code in setOf(11, 14) -> "序列号不在有效期内"
        else -> state.message.ifBlank { "请检查网络连接和序列号" }
    }

    private sealed interface LiveMode {
        data object Idle : LiveMode
        data class Enroll(val name: String) : LiveMode
        data object Identify : LiveMode
    }

    private companion object {
        const val REQUEST_ENROLL = 1001
        const val REQUEST_IDENTIFY = 1002
        const val MATCH_THRESHOLD = 0.80f
        const val DEFAULT_SERIAL_NUMBER = "E3AU-XMJ2-SRMS-HJTZ"
        const val ACTIVATION_CALLBACK_TIMEOUT_MS = 8_000L
        const val LIVE_IDENTIFY_INTERVAL_MS = 700L
        val SERIAL_NUMBER_PATTERN = Regex("[A-Z0-9]{4}(-[A-Z0-9]{4}){3}")
    }
}