package com.example.rknncamera

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.net.Uri
import android.provider.MediaStore
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.ExifInterface
import android.media.ImageReader
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import android.graphics.Color
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class MainActivity : Activity() {
    private lateinit var preview: TextureView
    private lateinit var overlay: DetectionOverlay
    private lateinit var stillImage: ImageView
    private lateinit var stats: TextView
    private lateinit var detectionResults: TextView
    private lateinit var inferenceStatusIndicator: TextView
    private lateinit var manager: CameraManager
    private val imageThread = HandlerThread("ImageReaderThread").apply { start() }
    private val imageHandler = Handler(imageThread.looper)
    private val inferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "RknnInferenceThread")
    }
    private val imageInferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "RknnImageInferenceThread")
    }
    private val detectorLock = Any()
    private val inferenceQueue = ArrayBlockingQueue<YuvFrame>(1)
    private val recentInferenceClasses = ArrayDeque<Int>(3)
    private val overlayUpdatePosted = AtomicBoolean(false)
    private val latestDetectionVersion = AtomicLong(0L)
    private val isDestroyed = AtomicBoolean(false)
    @Volatile private var imageMode = false
    @Volatile private var latestDetections: List<Detection> = emptyList()
    @Volatile private var captureFps = 0.0
    @Volatile private var inferenceFps = 0.0
    @Volatile private var lastInferenceMs = 0.0
    @Volatile private var lastDetectionCount = 0
    @Volatile private var latestDetectionFinishedMs = 0L
    private var captureLogFrames = 0
    private var inferenceLogFrames = 0
    @Volatile private var latestDetectionText = "{\n  \"count\": 0,\n  \"detections\": []\n}"
    private var statsWindowStartMs = SystemClock.elapsedRealtime()
    private var framesSinceStats = 0
    private var inferenceFramesSinceStats = 0
    private var inferenceStatsWindowStartMs = SystemClock.elapsedRealtime()
    private var lastUiUpdateMs = 0L
    private var lastDetectionUiUpdateMs = 0L
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var detector: RknnDetector? = null

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        stillImage = findViewById(R.id.still_image)
        stats = findViewById(R.id.stats)
        detectionResults = findViewById(R.id.detection_results)
        inferenceStatusIndicator = findViewById(R.id.inference_status_indicator)
        findViewById<android.view.View>(R.id.select_image).setOnClickListener { openImagePicker(compareYuv = false) }
        findViewById<android.view.View>(R.id.select_image_yuv).setOnClickListener { openImagePicker(compareYuv = true) }
        findViewById<android.view.View>(R.id.save_realtime_frame).setOnClickListener { requestSaveRealtimeFrame() }
        preview.scaleX = 1f
        overlay.visibility = android.view.View.VISIBLE
        detector = RknnDetector(assets, "helmet.sanitized-rk3568.rknn")
        manager = getSystemService(CameraManager::class.java)
        startInferenceWorker()
        preview.surfaceTextureListener = textureListener
    }

    @Volatile private var latestCameraFrame: YuvFrame? = null
    private var compareYuvImage = false

    private fun requestSaveRealtimeFrame() {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_STORAGE
            )
            return
        }
        saveLatestRealtimeFrame()
    }

    private fun saveLatestRealtimeFrame() {
        val frame = latestCameraFrame
        if (frame == null) {
            Toast.makeText(this, "暂时没有可保存的实时帧", Toast.LENGTH_SHORT).show()
            return
        }
        stats.text = "状态：正在保存实时帧..."
        imageInferenceExecutor.execute {
            val bitmap = try {
                RknnDetector.yuvFrameToBitmap(frame)
            } catch (error: RuntimeException) {
                Log.e(LOG_TAG, "convert realtime frame to bitmap failed", error)
                null
            }
            if (bitmap == null) {
                runOnUiThread { stats.text = "状态：实时帧转换失败" }
                return@execute
            }

            val resolver = contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "rknn_realtime_${System.currentTimeMillis()}.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/RKNN Camera")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri == null) {
                bitmap.recycle()
                runOnUiThread { stats.text = "状态：无法创建相册文件" }
                return@execute
            }

            val saved = try {
                resolver.openOutputStream(uri)?.use { output ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)
                } == true
            } catch (error: Exception) {
                Log.e(LOG_TAG, "save realtime frame failed", error)
                false
            } finally {
                bitmap.recycle()
            }

            if (saved && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                resolver.update(
                    uri,
                    ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) },
                    null,
                    null
                )
            } else if (!saved) {
                resolver.delete(uri, null, null)
            }
            runOnUiThread {
                stats.text = if (saved) "状态：实时帧已保存到相册\n尺寸：${frame.width} × ${frame.height}" else "状态：实时帧保存失败"
                Toast.makeText(this, if (saved) "实时帧已保存到相册" else "实时帧保存失败", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun openImagePicker(compareYuv: Boolean) {
        compareYuvImage = compareYuv
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "image/*"
            },
            REQUEST_IMAGE
        )
    }

    @Deprecated("Uses the Activity result API available in this legacy Activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                if (compareYuvImage) inferSelectedImageRgbAndYuv(uri) else inferSelectedImage(uri)
            }
        }
    }

    private fun inferSelectedImageRgbAndYuv(uri: Uri) {
        imageMode = true
        session?.stopRepeating()
        inferenceQueue.clear()
        runOnUiThread {
            stillImage.visibility = android.view.View.VISIBLE
            preview.visibility = android.view.View.GONE
            preview.rotation = 0f
            overlay.rotation = 0f
            stats.text = "状态：RGB/YUV 对比推理中..."
        }
        imageInferenceExecutor.execute {
            val bitmap = decodeOrientedBitmap(uri)
            if (bitmap == null) {
                runOnUiThread { stats.text = "状态：图片读取失败" }
                return@execute
            }
            val startedNs = SystemClock.elapsedRealtimeNanos()
            val result = synchronized(detectorLock) {
                detector?.detectRgbAndYuv(bitmap) ?: RgbYuvDetections(emptyList(), emptyList())
            }
            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000.0
           // val json = formatComparisonJson(result)
            runOnUiThread {
                if (!isDestroyed.get()) {
                    stillImage.setImageBitmap(bitmap)
                    overlay.setImageSourceSize(bitmap.width, bitmap.height)
                    overlay.setComparisonDetections(result.rgb, result.yuv)
                   // detectionResults.text = json
                    stats.text = "状态：RGB/YUV 对比推理完成\n总耗时：${"%.1f".format(Locale.US, elapsedMs)} ms\nRGB 检测数量：${result.rgb.size}\nYUV 检测数量：${result.yuv.size}\n输入尺寸：${bitmap.width} × ${bitmap.height}"
                }
            }
        }
    }

//    private fun formatComparisonJson(result: RgbYuvDetections): String =
//        """{
//            "rgb": ${formatDetectionJson(result.rgb)},
//            "yuv": ${formatDetectionJson(result.yuv)}
//        }""".trimIndent()

    private fun inferSelectedImage(uri: Uri) {
        imageMode = true
        session?.stopRepeating()
        inferenceQueue.clear()
        runOnUiThread {
            stillImage.visibility = android.view.View.VISIBLE
            preview.visibility = android.view.View.GONE
            preview.rotation = 0f
            overlay.rotation = 0f
            stats.text = "状态：图片推理中..."
        }
        imageInferenceExecutor.execute {
            val bitmap = decodeOrientedBitmap(uri)
            if (bitmap == null) {
                runOnUiThread { stats.text = "状态：图片读取失败" }
                return@execute
            }
            val startedNs = SystemClock.elapsedRealtimeNanos()
            val result = synchronized(detectorLock) {
                detector?.detect(bitmap).orEmpty()
            }
            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000.0
           // val json = formatDetectionJson(result)
            latestDetections = result.toList()
            lastDetectionCount = result.size
           // latestDetectionText = json
            runOnUiThread {
                if (!isDestroyed.get()) {
                    stillImage.setImageBitmap(bitmap)
                    overlay.setImageSourceSize(bitmap.width, bitmap.height)
                    overlay.setImageMode()
                   // detectionResults.text = json
                    overlay.setDetections(result)
                    stats.text = "状态：图片推理完成\n推理耗时：${"%.1f".format(Locale.US, elapsedMs)} ms\n检测数量：${result.size}\n输入尺寸：${bitmap.width} × ${bitmap.height}"
                }
            }
        }
    }

    private fun decodeOrientedBitmap(uri: Uri): android.graphics.Bitmap? {
        val bitmap = contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream) ?: return null
        val orientation = contentResolver.openInputStream(uri)?.use {
            ExifInterface(it).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } ?: ExifInterface.ORIENTATION_NORMAL
        if (orientation == ExifInterface.ORIENTATION_NORMAL) return bitmap

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
            else -> return bitmap
        }
        return android.graphics.Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        ).also {
            if (it !== bitmap) bitmap.recycle()
        }
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, w: Int, h: Int) = openCamera()
        override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) = Unit
        override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
        override fun onSurfaceTextureUpdated(s: SurfaceTexture) = Unit
    }

    private fun openCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
            return
        }
        val id = manager.cameraIdList.firstOrNull { manager.getCameraCharacteristics(it).get(android.hardware.camera2.CameraCharacteristics.LENS_FACING) == android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK } ?: manager.cameraIdList[0]
        manager.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) { camera = device; createSession(device) }
            override fun onDisconnected(device: CameraDevice) { device.close() }
            override fun onError(device: CameraDevice, error: Int) { device.close(); camera = null }
        }, null)
    }

    private fun createSession(device: CameraDevice) {
        val texture = preview.surfaceTexture ?: return
        texture.setDefaultBufferSize(CAMERA_WIDTH, CAMERA_HEIGHT)
        val display = Surface(texture)
        // ImageReader 使用与模型训练常见比例一致的相机帧，随后在 native 层统一 letterbox 到 640×640。
        reader = ImageReader.newInstance(CAMERA_WIDTH, CAMERA_HEIGHT, ImageFormat.YUV_420_888, 2).also {
            it.setOnImageAvailableListener({ source ->
                source.acquireLatestImage()?.use { image ->
                    val copyStartNs = SystemClock.elapsedRealtimeNanos()
                    val rawFrame = RknnDetector.copyFrame(image)
                    val frame = RknnDetector.rotate180(rawFrame)
                    latestCameraFrame = frame
                    val copyMs = (SystemClock.elapsedRealtimeNanos() - copyStartNs) / 1_000_000.0
                    framesSinceStats++
                    captureLogFrames++
                    if (captureLogFrames % LOG_EVERY_N_FRAMES == 0) {
                        Log.i(LOG_TAG, "capture frame=$captureLogFrames size=${image.width}x${image.height} copyMs=${"%.1f".format(Locale.US, copyMs)} queue=${inferenceQueue.size}")
                    }
                    offerLatestFrame(frame)
                    updateCaptureStats(image.width, image.height)
                }
            }, imageHandler)
        }


        device.createCaptureSession(listOf(display, reader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(value: CameraCaptureSession) {
                session = value
                runOnUiThread {
                    if (!isDestroyed.get()) {
                        overlay.setRealtimeSourceSize(CAMERA_WIDTH, CAMERA_HEIGHT)
                    }
                }
                val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(display); addTarget(reader!!.surface)
                }.build()
                value.setRepeatingRequest(request, null, null)
            }
            override fun onConfigureFailed(value: CameraCaptureSession) = Toast.makeText(this@MainActivity, "Camera2 配置失败", Toast.LENGTH_SHORT).show()
        }, null)
//        device.createCaptureSession(listOf(display, reader!!.surface), object : CameraCaptureSession.StateCallback() {
//            override fun onConfigured(value: CameraCaptureSession) {
//                session = value
//                val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
//                    addTarget(display); addTarget(reader!!.surface)
//                }.build()
//                value.setRepeatingRequest(request, null, null)
//            }
//            override fun onConfigureFailed(value: CameraCaptureSession) = Toast.makeText(this@MainActivity, "Camera2 配置失败", Toast.LENGTH_SHORT).show()
//        }, null)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_STORAGE && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            saveLatestRealtimeFrame()
            return
        }
        if (requestCode == REQUEST_CAMERA && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) openCamera()
    }

    private fun startInferenceWorker() {
        inferenceExecutor.execute {
            while (!isDestroyed.get()) {
                val frame = try {
                    inferenceQueue.take()
                } catch (_: InterruptedException) {
                    break
                }

                val startNs = SystemClock.elapsedRealtimeNanos()
                val result = synchronized(detectorLock) {
                    detector?.detect(frame).orEmpty()
                }
                val finishedMs = SystemClock.elapsedRealtime()
                lastInferenceMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
                latestDetectionFinishedMs = finishedMs
                inferenceLogFrames++
                if (inferenceLogFrames % LOG_EVERY_N_FRAMES == 0) {
                    Log.i(LOG_TAG, "inference frame=$inferenceLogFrames totalMs=${"%.1f".format(Locale.US, lastInferenceMs)} resultCount=${result.size}")
                }
                latestDetections = result.toList()
                lastDetectionCount = result.size
               // latestDetectionText = formatDetectionJson(result)
                updateInferenceStatus(result)
                latestDetectionVersion.incrementAndGet()
                publishLatestDetections()
                inferenceFramesSinceStats++

                val nowMs = SystemClock.elapsedRealtime()
                val elapsedMs = nowMs - inferenceStatsWindowStartMs
                if (elapsedMs >= UI_UPDATE_INTERVAL_MS) {
                    inferenceFps = inferenceFramesSinceStats * 1000.0 / elapsedMs
                    inferenceFramesSinceStats = 0
                    inferenceStatsWindowStartMs = nowMs
                }

                // 每个推理结果都提交对应的绘制快照，保证 overlay 与推理帧一一对应。
            }
        }
    }

    private fun updateInferenceStatus(detections: List<Detection>) {
        val statusClass = when {
            detections.any { it.classId == HELMET_CLASS_ID } -> HELMET_CLASS_ID
            detections.isNotEmpty() && detections.all { it.classId == HEAD_CLASS_ID } -> HEAD_CLASS_ID
            else -> UNKNOWN_CLASS_ID
        }
        if (recentInferenceClasses.size == STATUS_WINDOW_SIZE) {
            recentInferenceClasses.removeFirst()
        }
        recentInferenceClasses.addLast(statusClass)
    }

    private fun renderInferenceStatus() {
        val recentClasses = recentInferenceClasses.toList()
        val hasHelmet = recentClasses.any { it == HELMET_CLASS_ID }
        val allHeads = recentClasses.size == STATUS_WINDOW_SIZE &&
            recentClasses.all { it == HEAD_CLASS_ID }
        when {
            hasHelmet -> {
                inferenceStatusIndicator.setBackgroundColor(Color.rgb(198, 40, 40))
                inferenceStatusIndicator.text = "最近三次推理状态：检测到 helmet"
            }
            allHeads -> {
                inferenceStatusIndicator.setBackgroundColor(Color.rgb(46, 125, 50))
                inferenceStatusIndicator.text = "最近三次推理状态：全部为 head"
            }
            else -> {
                inferenceStatusIndicator.setBackgroundColor(Color.rgb(89, 99, 107))
                inferenceStatusIndicator.text = "最近三次推理状态：等待 head 连续三次"
            }
        }
    }

    private fun publishLatestDetections() {
        if (!overlayUpdatePosted.compareAndSet(false, true)) return

        runOnUiThread {
            val renderedVersion = latestDetectionVersion.get()
            if (!isDestroyed.get()) {
                val publishDelayMs = SystemClock.elapsedRealtime() - latestDetectionFinishedMs
                Log.i(LOG_TAG, "ui publish delayMs=$publishDelayMs detections=${latestDetections.size}")
                detectionResults.text = latestDetectionText
                overlay.setDetections(latestDetections)
                renderInferenceStatus()
            }
            overlayUpdatePosted.set(false)
            if (!isDestroyed.get() && latestDetectionVersion.get() != renderedVersion) {
                publishLatestDetections()
            }
        }
    }

//    private fun formatDetectionJson(items: List<Detection>): String {
//        val detections = items.mapIndexed { index, item ->
//            val className = when (item.classId) {
//                0 -> "head"
//                1 -> "helmet"
//                else -> "class-${item.classId}"
//            }
//            val confidence = item.score.coerceIn(0f, 1f)
//
//            """{
//                "index": ${index + 1},
//                "class_id": ${item.classId},
//                "class_name": "$className",
//                "confidence": ${"%.6f".format(Locale.US, confidence)},
//                "bbox": {
//                    "left": ${"%.6f".format(Locale.US, item.left)},
//                    "top": ${"%.6f".format(Locale.US, item.top)},
//                    "right": ${"%.6f".format(Locale.US, item.right)},
//                    "bottom": ${"%.6f".format(Locale.US, item.bottom)}
//                }
//            }""".trimIndent()
//        }
//
//        return """{
//            "count": ${items.size},
//            "detections": [${detections.joinToString(",")}]
//        }""".trimIndent()
//    }

    private fun offerLatestFrame(frame: YuvFrame) {
        if (!inferenceQueue.offer(frame)) {
            inferenceQueue.poll()
            inferenceQueue.offer(frame)
        }
    }

    private fun updateCaptureStats(width: Int, height: Int) {
        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastUiUpdateMs < UI_UPDATE_INTERVAL_MS) return

        val elapsedMs = nowMs - statsWindowStartMs
        if (elapsedMs > 0) {
            captureFps = framesSinceStats * 1000.0 / elapsedMs
        }
        val sampledFrames = framesSinceStats
        framesSinceStats = 0
        statsWindowStartMs = nowMs
        lastUiUpdateMs = nowMs

        val text = "状态：RKNN 异步推理中（检测框绘制已开启）\n\n" +
            "预览帧率：${"%.1f".format(Locale.getDefault(), captureFps)} FPS\n" +
            "推理帧率：${"%.1f".format(Locale.getDefault(), inferenceFps)} FPS\n" +
            "推理耗时：${"%.1f".format(Locale.getDefault(), lastInferenceMs)} ms\n" +
            "检测数量：$lastDetectionCount\n" +
            "采样帧数：$sampledFrames\n" +
            "输入尺寸：$width × $height\n" +
            "颜色格式：YUV_420_888"
        runOnUiThread { stats.text = text }
    }

    override fun onDestroy() {
        isDestroyed.set(true)
        session?.close()
        camera?.close()
        reader?.close()
        inferenceExecutor.shutdownNow()
        imageInferenceExecutor.shutdownNow()
        try {
            inferenceExecutor.awaitTermination(1, TimeUnit.SECONDS)
            imageInferenceExecutor.awaitTermination(1, TimeUnit.SECONDS)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        inferenceQueue.clear()
        detector?.close()
        imageThread.quitSafely()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_CAMERA = 1001
        private const val REQUEST_IMAGE = 1002
        private const val REQUEST_STORAGE = 1003
        private const val CAMERA_WIDTH = 640
        private const val CAMERA_HEIGHT = 480
        private const val UI_UPDATE_INTERVAL_MS = 100L
        private const val JSON_UPDATE_INTERVAL_MS = 500L
        private const val LOG_EVERY_N_FRAMES = 30
        private const val STATUS_WINDOW_SIZE = 3
        private const val HEAD_CLASS_ID = 0
        private const val HELMET_CLASS_ID = 1
        private const val UNKNOWN_CLASS_ID = -1
        private const val LOG_TAG = "RknnPipeline"
    }
}