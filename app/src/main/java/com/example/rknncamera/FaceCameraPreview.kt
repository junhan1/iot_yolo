package com.example.rknncamera

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

/** Camera2 预览与帧采样。只在 [requestFrame] 后交付一帧，避免积压视频帧。 */
class FaceCameraPreview(
    private val activity: Activity,
    private val preview: TextureView,
    private val frameExecutor: Executor,
    private val onFrame: (android.graphics.Bitmap) -> Unit,
    private val onError: (String) -> Unit
) {
    private val cameraManager = activity.getSystemService(CameraManager::class.java)
    private val cameraThread = HandlerThread("FaceCamera").apply { start() }
    private val cameraHandler = Handler(cameraThread.looper)
    private val frameRequested = AtomicBoolean(false)
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null

    fun start() {
        preview.surfaceTextureListener = surfaceListener
        if (preview.isAvailable) openCamera()
    }

    fun requestFrame(): Boolean = frameRequested.compareAndSet(false, true)

    fun close() {
        frameRequested.set(false)
        session?.close()
        camera?.close()
        reader?.close()
        cameraThread.quitSafely()
    }

    private val surfaceListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) = openCamera()
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
    }

    private fun openCamera() {
        if (activity.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            activity.requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
            return
        }
        val id = cameraManager.cameraIdList.firstOrNull {
            cameraManager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT
        } ?: cameraManager.cameraIdList.firstOrNull()
        if (id == null) {
            onError("未找到可用摄像头")
            return
        }
        cameraManager.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                camera = device
                createSession(device)
            }
            override fun onDisconnected(device: CameraDevice) {
                device.close()
                camera = null
            }
            override fun onError(device: CameraDevice, error: Int) {
                device.close()
                camera = null
                onError("摄像头打开失败：$error")
            }
        }, cameraHandler)
    }

    private fun createSession(device: CameraDevice) {
        val texture = preview.surfaceTexture ?: return
        texture.setDefaultBufferSize(FRAME_WIDTH, FRAME_HEIGHT)
        val display = Surface(texture)
        reader = ImageReader.newInstance(FRAME_WIDTH, FRAME_HEIGHT, ImageFormat.YUV_420_888, 2).also { imageReader ->
            imageReader.setOnImageAvailableListener({ source ->
                source.acquireLatestImage()?.use { image ->
                    if (!frameRequested.compareAndSet(true, false)) return@use
                    val frame = RknnDetector.copyFrame(image)
                    frameExecutor.execute {
                        onFrame(RknnDetector.yuvFrameToBitmap(frame))
                    }
                }
            }, cameraHandler)
        }
        device.createCaptureSession(listOf(display, reader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(value: CameraCaptureSession) {
                session = value
                val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(display)
                    addTarget(reader!!.surface)
                }.build()
                value.setRepeatingRequest(request, null, cameraHandler)
            }
            override fun onConfigureFailed(value: CameraCaptureSession) = onError("摄像头预览配置失败")
        }, cameraHandler)
    }

    companion object {
        const val REQUEST_CAMERA = 2001
        private const val FRAME_WIDTH = 640
        private const val FRAME_HEIGHT = 480
    }
}