package com.example.rknncamera

import android.Manifest
import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.ImageReader
import android.view.Surface
import android.view.TextureView
import android.widget.TextView
import android.widget.Toast
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {
    private lateinit var preview: TextureView
    private lateinit var overlay: DetectionOverlay
    private lateinit var stats: TextView
    private lateinit var manager: CameraManager
    private val imageThread = HandlerThread("ImageReaderThread").apply { start() }
    private val imageHandler = Handler(imageThread.looper)
    private val inferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "RknnInferenceThread")
    }
    private val inferenceQueue = ArrayBlockingQueue<YuvFrame>(1)
    private val isDestroyed = AtomicBoolean(false)
    @Volatile private var captureFps = 0.0
    @Volatile private var inferenceFps = 0.0
    @Volatile private var lastInferenceMs = 0.0
    @Volatile private var lastDetectionCount = 0
    private var statsWindowStartMs = SystemClock.elapsedRealtime()
    private var framesSinceStats = 0
    private var inferenceFramesSinceStats = 0
    private var inferenceStatsWindowStartMs = SystemClock.elapsedRealtime()
    private var lastUiUpdateMs = 0L
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var detector: RknnDetector? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        stats = findViewById(R.id.stats)
        preview.scaleX = -1f
        overlay.visibility = android.view.View.GONE
        detector = RknnDetector(assets, "helmet.sanitized-rk3568.rknn")
        manager = getSystemService(CameraManager::class.java)
        startInferenceWorker()
        preview.surfaceTextureListener = textureListener
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
        texture.setDefaultBufferSize(640, 640)
        val display = Surface(texture)
        //创建图像读取器  图像格式YUV_420_888  输出图像宽高 640*640  AI模型要求输入该尺寸  maxImages：最大缓存帧数（ImageReader内部最多同时持有2张图片，防止内存暴涨）
        reader = ImageReader.newInstance(640, 640, ImageFormat.YUV_420_888, 2).also {
            it.setOnImageAvailableListener({ source ->
                source.acquireLatestImage()?.use { image ->
                    val frame = RknnDetector.copyFrame(image)
                    framesSinceStats++
                    offerLatestFrame(frame)
                    updateCaptureStats(image.width, image.height)
                }
            }, imageHandler)
        }


        device.createCaptureSession(listOf(display, reader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(value: CameraCaptureSession) {
                session = value
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
                val result = detector?.detect(frame).orEmpty()
                lastInferenceMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
                lastDetectionCount = result.size
                inferenceFramesSinceStats++

                val nowMs = SystemClock.elapsedRealtime()
                val elapsedMs = nowMs - inferenceStatsWindowStartMs
                if (elapsedMs >= UI_UPDATE_INTERVAL_MS) {
                    inferenceFps = inferenceFramesSinceStats * 1000.0 / elapsedMs
                    inferenceFramesSinceStats = 0
                    inferenceStatsWindowStartMs = nowMs
                }

                // 结果当前不绘制；如果启用框绘制，应只把最新结果提交到主线程。
            }
        }
    }

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

        val text = "状态：RKNN 异步推理中（检测框绘制已关闭）\n\n" +
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
        try {
            inferenceExecutor.awaitTermination(1, TimeUnit.SECONDS)
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
        private const val UI_UPDATE_INTERVAL_MS = 100L
    }
}