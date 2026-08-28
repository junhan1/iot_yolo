package com.example.rknncamera

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.graphics.ImageFormat
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.AttributeSet
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/** HMI 首页：真实相机预览、RKNN 异步推理和状态指标的组合页面。 */
class MainActivity1 : Activity() {
    private lateinit var preview: TextureView
    private lateinit var overlay: HmiDetectionOverlay
    private lateinit var resultBadge: ViewGroup
    private lateinit var resultSymbol: TextView
    private lateinit var resultText: TextView
    private lateinit var warningCard: ViewGroup
    private lateinit var warningText: TextView
    private lateinit var statusLine: TextView
    private lateinit var statusIndicator: View
    private lateinit var previewFps: TextView
    private lateinit var inferenceFps: TextView
    private lateinit var inferenceTime: TextView
    private lateinit var detectionCount: TextView
    private lateinit var sampleFrames: TextView
    private lateinit var inputSize: TextView
    private lateinit var manager: CameraManager
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var detector: RknnDetector? = null
    private val imageThread = HandlerThread("HmiImageReader").apply { start() }
    private val imageHandler = Handler(imageThread.looper)
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val inferenceQueue = ArrayBlockingQueue<YuvFrame>(1)
    private val detectorLock = Any()
    private val destroyed = AtomicBoolean(false)
    private val resultUpdatePosted = AtomicBoolean(false)
    @Volatile private var latestDetections: List<Detection> = emptyList()
    @Volatile private var lastInferenceMs = 0.0
    private val recentInferenceStates = ArrayDeque<InferenceState>(INFERENCE_WINDOW_SIZE)

    private enum class InferenceState {
        HEAD,
        HELMET,
        NO_DETECTION
    }
    private var captureFrames = 0
    private var inferenceFrames = 0
    private var sampleFrameCount = 0
    private var statsStartedMs = SystemClock.elapsedRealtime()
    private var lastUiMs = 0L
    private var captureRate = 0.0
    private var inferenceRate = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main1)
        preview = findViewById(R.id.camera_preview)
        overlay = findViewById(R.id.hmi_overlay)
        resultBadge = findViewById(R.id.result_badge)
        resultBadge.visibility = View.GONE
        resultSymbol = findViewById(R.id.result_symbol)
        resultText = findViewById(R.id.result_text)
        warningCard = findViewById(R.id.warning_card)
        warningText = findViewById(R.id.warning_text)
        statusLine = findViewById(R.id.status_line)
        statusIndicator = findViewById(R.id.status_indicator)
        previewFps = findViewById(R.id.preview_fps)
        inferenceFps = findViewById(R.id.inference_fps)
        inferenceTime = findViewById(R.id.inference_time)
        detectionCount = findViewById(R.id.detection_count)
        sampleFrames = findViewById(R.id.sample_frames)
        inputSize = findViewById(R.id.input_size)
        manager = getSystemService(CameraManager::class.java)
        detector = RknnDetector(assets, "helmet.sanitized-rk3568.rknn")
        startInferenceWorker()
        preview.surfaceTextureListener = textureListener
    }

    override fun onStart() {
        super.onStart()
        destroyed.set(false)
        if (preview.isAvailable) openCamera()
    }

    override fun onStop() {
        closeCamera()
        super.onStop()
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) = openCamera()
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean { closeCamera(); return true }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
    }

    private fun openCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
            return
        }
        val cameraId = manager.cameraIdList.firstOrNull { id ->
            manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        } ?: manager.cameraIdList.firstOrNull() ?: return
        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) { camera = device; createCaptureSession(device) }
            override fun onDisconnected(device: CameraDevice) { device.close(); if (camera === device) camera = null }
            override fun onError(device: CameraDevice, error: Int) { device.close(); if (camera === device) camera = null }
        }, null)
    }

    private fun createCaptureSession(device: CameraDevice) {
        val texture = preview.surfaceTexture ?: return
        texture.setDefaultBufferSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        val display = Surface(texture)
        reader?.close()
        reader = ImageReader.newInstance(CAMERA_WIDTH, CAMERA_HEIGHT, ImageFormat.YUV_420_888, 2).also { source ->
            source.setOnImageAvailableListener({ imageReader ->
                imageReader.acquireLatestImage()?.use { image ->
                    val frame = RknnDetector.rotate180(RknnDetector.copyFrame(image))
                    captureFrames++
                    sampleFrameCount++
                    offerLatestFrame(frame)
                    updateStats(image.width, image.height)
                }
            }, imageHandler)
        }
        device.createCaptureSession(listOf(display, reader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(value: CameraCaptureSession) {
                if (destroyed.get()) { value.close(); return }
                session = value
                overlay.setRealtimeSourceSize(CAMERA_WIDTH, CAMERA_HEIGHT)
                val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(display)
                    addTarget(reader!!.surface)
                }.build()
                value.setRepeatingRequest(request, null, null)
            }
            override fun onConfigureFailed(value: CameraCaptureSession) {
                Toast.makeText(this@MainActivity1, "Camera2 配置失败", Toast.LENGTH_SHORT).show()
            }
        }, null)
    }

    private fun startInferenceWorker() {
        inferenceExecutor.execute {
            while (!destroyed.get()) {
                val frame = try {
                    inferenceQueue.take()
                } catch (_: InterruptedException) {
                    break
                }
                try {
                    val startedNs = SystemClock.elapsedRealtimeNanos()
                    val detections = synchronized(detectorLock) { detector?.detect(frame).orEmpty() }
                    lastInferenceMs = (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000.0
                    latestDetections = detections
                    inferenceFrames++
                    updateRates()
                    publishLatestResult()
                } catch (error: Exception) {
                    Log.e(LOG_TAG, "RKNN inference failed; continuing with the next frame", error)
                    publishInferenceError()
                }
            }
        }
    }

    private fun updateRates() {
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - statsStartedMs
        if (elapsed < STATS_WINDOW_MS) return

        captureRate = captureFrames * 1000.0 / elapsed
        inferenceRate = inferenceFrames * 1000.0 / elapsed
        captureFrames = 0
        inferenceFrames = 0
        statsStartedMs = now
    }

    private fun publishLatestResult() {
        if (!resultUpdatePosted.compareAndSet(false, true)) return

        runOnUiThread {
            resultUpdatePosted.set(false)
            if (!destroyed.get()) renderDetections(latestDetections)
        }
    }

    private fun publishInferenceError() {
        if (resultUpdatePosted.compareAndSet(false, true)) {
            runOnUiThread {
                resultUpdatePosted.set(false)
                if (!destroyed.get()) {
                    statusLine.text = "状态：推理异常，正在重试下一帧"
                }
            }
        }
    }

    private fun renderDetections(detections: List<Detection>) {
        val hasHelmet = detections.any { it.classId == HELMET_CLASS_ID }
        val hasHead = detections.any { it.classId == HEAD_CLASS_ID }
        val hasKnownDetection = hasHelmet || hasHead
        val safe = hasHelmet && !hasHead
        val accent = if (safe) SAFE_GREEN else DANGER_RED

        overlay.setDetections(detections)
        resultBadge.visibility = if (hasKnownDetection) View.VISIBLE else View.GONE
        if (hasKnownDetection) {
            resultBadge.setBackgroundResource(if (safe) R.drawable.bg_hmi_safe_badge else R.drawable.bg_hmi_danger_badge)
            resultSymbol.setBackgroundColor(accent)
            resultSymbol.setTextColor(if (safe) DARK_GREEN else Color.WHITE)
            resultSymbol.text = if (safe) "✓" else "!"
            resultText.text = if (safe) "已佩戴安全帽" else "未佩戴安全帽"
            resultText.setTextColor(if (safe) LIGHT_GREEN else LIGHT_RED)
        }
        updateRecentInferenceStates(detections)
        renderRecentInferenceStatus()
        updateMetrics(detections)
    }

    private fun updateRecentInferenceStates(detections: List<Detection>) {
        val state = when {
            detections.any { it.classId == HELMET_CLASS_ID } -> InferenceState.HELMET
            detections.any { it.classId == HEAD_CLASS_ID } -> InferenceState.HEAD
            else -> InferenceState.NO_DETECTION
        }
        if (recentInferenceStates.size == INFERENCE_WINDOW_SIZE) recentInferenceStates.removeFirst()
        recentInferenceStates.addLast(state)
    }

    private fun renderRecentInferenceStatus() {
        val hasHelmet = recentInferenceStates.any { it == InferenceState.HELMET }
        val hasCompleteWindow = recentInferenceStates.size == INFERENCE_WINDOW_SIZE
        val allNoDetections = hasCompleteWindow &&
            recentInferenceStates.all { it == InferenceState.NO_DETECTION }
        when {
            hasHelmet -> {
                warningCard.setBackgroundResource(R.drawable.bg_hmi_notice_white)
                warningText.text = "安全帽已佩戴"
                warningText.setTextColor(Color.WHITE)
            }
            allNoDetections -> {
                warningCard.setBackgroundResource(R.drawable.bg_hmi_warning)
                warningText.text = "未检测到"
                warningText.setTextColor(LIGHT_RED)
            }
            hasCompleteWindow -> {
                warningCard.setBackgroundResource(R.drawable.bg_hmi_warning)
                warningText.text = "未佩戴安全帽"
                warningText.setTextColor(LIGHT_RED)
            }
            else -> {
                warningCard.setBackgroundResource(R.drawable.bg_hmi_status)
                warningText.text = "正在等待最近三次推理结果"
                warningText.setTextColor(Color.WHITE)
            }
        }
    }

    private fun updateMetrics(detections: List<Detection>) {
        previewFps.text = "预览帧率\n${formatMetric(captureRate)} FPS"
        inferenceFps.text = "推理帧率\n${formatMetric(inferenceRate)} FPS"
        inferenceTime.text = "推理耗时\n${formatMetric(lastInferenceMs)} ms"
        detectionCount.text = "检测数量\n${detections.size}"
    }

    private fun updateStats(width: Int, height: Int) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastUiMs < UI_UPDATE_INTERVAL_MS) return
        lastUiMs = now
        runOnUiThread {
            if (destroyed.get()) return@runOnUiThread
            sampleFrames.text = "采样帧数                              $sampleFrameCount"
            inputSize.text = "输入尺寸                              $width × $height"
        }
    }

    private fun offerLatestFrame(frame: YuvFrame) {
        if (!inferenceQueue.offer(frame)) {
            inferenceQueue.poll()
            inferenceQueue.offer(frame)
        }
    }

    private fun closeCamera() {
        session?.close(); session = null
        camera?.close(); camera = null
        reader?.close(); reader = null
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) openCamera()
    }

    override fun onDestroy() {
        destroyed.set(true)
        closeCamera()
        inferenceQueue.clear()
        inferenceExecutor.shutdownNow()
        try { inferenceExecutor.awaitTermination(1, TimeUnit.SECONDS) } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
        synchronized(detectorLock) { detector?.close(); detector = null }
        imageThread.quitSafely()
        super.onDestroy()
    }

    private fun formatMetric(value: Double) = "%.1f".format(Locale.US, value)

    companion object {
        private const val REQUEST_CAMERA = 1001
        private const val CAMERA_WIDTH = 640
        private const val CAMERA_HEIGHT = 480
        private const val STATS_WINDOW_MS = 1000L
        private const val UI_UPDATE_INTERVAL_MS = 100L
        private const val HEAD_CLASS_ID = 0
        private const val HELMET_CLASS_ID = 1
        private const val INFERENCE_WINDOW_SIZE = 3
        private const val LOG_TAG = "MainActivity1"
        private val SAFE_GREEN = Color.rgb(99, 201, 106)
        private val DANGER_RED = Color.rgb(244, 67, 54)
        private val DARK_GREEN = Color.rgb(20, 32, 22)
        private val LIGHT_GREEN = Color.rgb(239, 249, 240)
        private val LIGHT_RED = Color.rgb(255, 116, 107)
    }
}

class HmiDetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private val density = resources.displayMetrics.density
    private val border = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(5f)
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = dp(18f)
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    }
    private val scan = Paint(Paint.ANTI_ALIAS_FLAG)
    private var detections: List<Detection> = emptyList()
    private var sourceWidth = 1f
    private var sourceHeight = 1f
    private var scanProgress = 0f

    fun setDetections(value: List<Detection>) {
        detections = value
        invalidate()
    }

    fun setRealtimeSourceSize(width: Int, height: Int) {
        sourceWidth = width.toFloat().coerceAtLeast(1f)
        sourceHeight = height.toFloat().coerceAtLeast(1f)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scale = maxOf(width / sourceWidth, height / sourceHeight)
        val offsetX = (sourceWidth * scale - width) / 2f
        val offsetY = (sourceHeight * scale - height) / 2f
        detections.forEach { item ->
            val left = item.left.coerceIn(0f, 1f) * sourceWidth * scale - offsetX
            val top = item.top.coerceIn(0f, 1f) * sourceHeight * scale - offsetY
            val right = item.right.coerceIn(0f, 1f) * sourceWidth * scale - offsetX
            val bottom = item.bottom.coerceIn(0f, 1f) * sourceHeight * scale - offsetY
            val color = if (item.classId == helmetClassId) helmetColor else headColor
            border.color = color
            canvas.drawRect(left, top, right, bottom, border)
            canvas.drawText(
                "${className(item.classId)} ${"%.0f".format(Locale.US, item.score * 100f)}%",
                left,
                (top - dp(8f)).coerceAtLeast(label.textSize),
                label
            )
        }

        scan.color = Color.argb(185, 255, 211, 53)
        val scanY = height * (.07f + scanProgress * .85f)
        canvas.drawRect(0f, scanY, width.toFloat(), scanY + dp(1f), scan)
        scanProgress = (scanProgress + .018f) % 1f
        postInvalidateDelayed(40L)
    }

    private fun className(classId: Int) = when (classId) {
        0 -> "head"
        1 -> "helmet"
        else -> "class-$classId"
    }

    private val helmetColor = Color.rgb(99, 201, 106)
    private val headColor = Color.rgb(244, 67, 54)
    private val helmetClassId = 1

    private fun dp(value: Float): Float = value * density
}