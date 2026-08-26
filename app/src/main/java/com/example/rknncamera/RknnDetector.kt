package com.example.rknncamera

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.media.Image
import java.nio.ByteBuffer
import java.nio.ByteOrder

class RknnDetector(assets: AssetManager, modelAssetName: String) : AutoCloseable {
    private var nativeHandle = nativeCreate(assets, modelAssetName)

    fun detect(frame: YuvFrame): List<Detection> {
        if (nativeHandle == 0L || frame.width <= 0 || frame.height <= 0) return emptyList()

        val scale = minOf(
            MODEL_INPUT_SIZE.toFloat() / frame.width,
            MODEL_INPUT_SIZE.toFloat() / frame.height
        )
        val resizedWidth = (frame.width * scale).toInt().coerceAtLeast(1)
        val resizedHeight = (frame.height * scale).toInt().coerceAtLeast(1)
        val padLeft = (MODEL_INPUT_SIZE - resizedWidth) / 2
        val padTop = (MODEL_INPUT_SIZE - resizedHeight) / 2

        return decodeDetections(
            nativeDetect(
                nativeHandle,
                frame.y,
                frame.u,
                frame.v,
                frame.width,
                frame.height,
                frame.yStride,
                frame.uvStride,
                frame.uvPixelStride
            )
        ).map { detection ->
            Detection(
                left = (detection.left * MODEL_INPUT_SIZE - padLeft) / resizedWidth,
                top = (detection.top * MODEL_INPUT_SIZE - padTop) / resizedHeight,
                right = (detection.right * MODEL_INPUT_SIZE - padLeft) / resizedWidth,
                bottom = (detection.bottom * MODEL_INPUT_SIZE - padTop) / resizedHeight,
                score = detection.score,
                classId = detection.classId
            )
        }.filter { it.right > 0f && it.bottom > 0f && it.left < 1f && it.top < 1f }
            .map { detection ->
                detection.copy(
                    left = detection.left.coerceIn(0f, 1f),
                    top = detection.top.coerceIn(0f, 1f),
                    right = detection.right.coerceIn(0f, 1f),
                    bottom = detection.bottom.coerceIn(0f, 1f)
                )
            }
    }

    /** 使用 YOLO letterbox 预处理，并把模型坐标还原为原图归一化坐标。 */
    fun detect(bitmap: Bitmap): List<Detection> {
        if (nativeHandle == 0L || bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) return emptyList()

        val scale = minOf(
            MODEL_INPUT_SIZE.toFloat() / bitmap.width,
            MODEL_INPUT_SIZE.toFloat() / bitmap.height
        )
        val resizedWidth = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val resizedHeight = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val padLeft = (MODEL_INPUT_SIZE - resizedWidth) / 2
        val padTop = (MODEL_INPUT_SIZE - resizedHeight) / 2
        val letterboxed = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888)

        return try {
            Canvas(letterboxed).apply {
                drawColor(Color.rgb(114, 114, 114))
                val resized = Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true)
                drawBitmap(resized, null, Rect(padLeft, padTop, padLeft + resizedWidth, padTop + resizedHeight), null)
                if (resized !== bitmap) resized.recycle()
            }

            val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
            letterboxed.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            val rgb = ByteBuffer
                .allocateDirect(pixels.size * RGB_CHANNELS)
                .order(ByteOrder.nativeOrder())
            pixels.forEach { pixel ->
                rgb.put((pixel shr 16 and 0xff).toByte())
                rgb.put((pixel shr 8 and 0xff).toByte())
                rgb.put((pixel and 0xff).toByte())
            }
            rgb.position(0)

            decodeDetections(nativeDetectRgb(nativeHandle, rgb, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
                .map { detection ->
                    Detection(
                        left = (detection.left * MODEL_INPUT_SIZE - padLeft) / resizedWidth,
                        top = (detection.top * MODEL_INPUT_SIZE - padTop) / resizedHeight,
                        right = (detection.right * MODEL_INPUT_SIZE - padLeft) / resizedWidth,
                        bottom = (detection.bottom * MODEL_INPUT_SIZE - padTop) / resizedHeight,
                        score = detection.score,
                        classId = detection.classId
                    )
                }
                .filter { it.right > 0f && it.bottom > 0f && it.left < 1f && it.top < 1f }
                .map { detection ->
                    detection.copy(
                        left = detection.left.coerceIn(0f, 1f),
                        top = detection.top.coerceIn(0f, 1f),
                        right = detection.right.coerceIn(0f, 1f),
                        bottom = detection.bottom.coerceIn(0f, 1f)
                    )
                }
        } finally {
            letterboxed.recycle()
        }
    }

    fun detectRgbAndYuv(bitmap: Bitmap): RgbYuvDetections {
        val yuvFrame = bitmapToYuvFrame(bitmap)
        return RgbYuvDetections(
            rgb = detect(bitmap),
            yuv = detect(yuvFrame)
        )
    }

    override fun close() {
        if (nativeHandle != 0L) nativeDestroy(nativeHandle).also { nativeHandle = 0L }
    }

    private fun decodeDetections(values: FloatArray): List<Detection> =
        values.toList().chunked(6).mapNotNull { item ->
            item.takeIf { it.size == 6 }?.let {
                Detection(it[0], it[1], it[2], it[3], it[4], it[5].toInt())
            }
        }

    private external fun nativeCreate(assets: AssetManager, modelAssetName: String): Long
    private external fun nativeDetect(
        handle: Long,
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        width: Int,
        height: Int,
        yStride: Int,
        uvStride: Int,
        uvPixelStride: Int
    ): FloatArray
    private external fun nativeDetectRgb(handle: Long, rgb: ByteBuffer, width: Int, height: Int): FloatArray
    private external fun nativeDestroy(handle: Long)

    companion object {
        private const val MODEL_INPUT_SIZE = 640
        private const val RGB_CHANNELS = 3

        init { System.loadLibrary("rknn_camera") }

        fun bitmapToYuvFrame(bitmap: Bitmap): YuvFrame {
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            val y = ByteBuffer.allocateDirect(width * height)
            val uvWidth = (width + 1) / 2
            val uvHeight = (height + 1) / 2
            val u = ByteBuffer.allocateDirect(uvWidth * uvHeight)
            val v = ByteBuffer.allocateDirect(uvWidth * uvHeight)

            fun rgbToYuv(red: Int, green: Int, blue: Int): Triple<Int, Int, Int> {
                val yValue = ((66 * red + 129 * green + 25 * blue + 128) shr 8) + 16
                val uValue = ((-38 * red - 74 * green + 112 * blue + 128) shr 8) + 128
                val vValue = ((112 * red - 94 * green - 18 * blue + 128) shr 8) + 128
                return Triple(yValue.coerceIn(0, 255), uValue.coerceIn(0, 255), vValue.coerceIn(0, 255))
            }

            for (row in 0 until height) {
                for (column in 0 until width) {
                    val pixel = pixels[row * width + column]
                    val yValue = rgbToYuv(pixel shr 16 and 0xff, pixel shr 8 and 0xff, pixel and 0xff).first
                    y.put(row * width + column, yValue.toByte())
                }
            }
            for (row in 0 until height step 2) {
                for (column in 0 until width step 2) {
                    var red = 0
                    var green = 0
                    var blue = 0
                    var count = 0
                    for (dy in 0..1) {
                        for (dx in 0..1) {
                            val sourceRow = (row + dy).coerceAtMost(height - 1)
                            val sourceColumn = (column + dx).coerceAtMost(width - 1)
                            val pixel = pixels[sourceRow * width + sourceColumn]
                            red += pixel shr 16 and 0xff
                            green += pixel shr 8 and 0xff
                            blue += pixel and 0xff
                            count++
                        }
                    }
                    val (_, uValue, vValue) = rgbToYuv(red / count, green / count, blue / count)
                    val uvIndex = (row / 2) * uvWidth + column / 2
                    u.put(uvIndex, uValue.toByte())
                    v.put(uvIndex, vValue.toByte())
                }
            }
            y.position(0)
            u.position(0)
            v.position(0)
            return YuvFrame(y, u, v, width, height, width, uvWidth, 1)
        }

        fun yuvFrameToBitmap(frame: YuvFrame): Bitmap {
            val pixels = IntArray(frame.width * frame.height)
            val yPlane = frame.y.duplicate()
            val uPlane = frame.u.duplicate()
            val vPlane = frame.v.duplicate()
            for (row in 0 until frame.height) {
                for (column in 0 until frame.width) {
                    val yValue = yPlane.get(row * frame.yStride + column).toInt() and 0xff
                    val uvOffset = (row / 2) * frame.uvStride + (column / 2) * frame.uvPixelStride
                    val uValue = (uPlane.get(uvOffset).toInt() and 0xff) - 128
                    val vValue = (vPlane.get(uvOffset).toInt() and 0xff) - 128
                    val yLimited = yValue - 16
                    val red = (298 * yLimited + 409 * vValue + 128) shr 8
                    val green = (298 * yLimited - 100 * uValue - 208 * vValue + 128) shr 8
                    val blue = (298 * yLimited + 516 * uValue + 128) shr 8
                    pixels[row * frame.width + column] = Color.rgb(
                        red.coerceIn(0, 255),
                        green.coerceIn(0, 255),
                        blue.coerceIn(0, 255)
                    )
                }
            }
            return Bitmap.createBitmap(pixels, frame.width, frame.height, Bitmap.Config.ARGB_8888)
        }

        fun rotate180(frame: YuvFrame): YuvFrame {
            val rotatedY = ByteBuffer.allocateDirect(frame.width * frame.height)
            for (row in 0 until frame.height) {
                for (column in 0 until frame.width) {
                    val sourceIndex = (frame.height - 1 - row) * frame.yStride + (frame.width - 1 - column)
                    rotatedY.put(row * frame.width + column, frame.y.get(sourceIndex))
                }
            }

            val uvWidth = (frame.width + 1) / 2
            val uvHeight = (frame.height + 1) / 2
            val rotatedU = ByteBuffer.allocateDirect(uvWidth * uvHeight)
            val rotatedV = ByteBuffer.allocateDirect(uvWidth * uvHeight)
            for (row in 0 until uvHeight) {
                for (column in 0 until uvWidth) {
                    val sourceRow = uvHeight - 1 - row
                    val sourceColumn = uvWidth - 1 - column
                    val sourceIndex = sourceRow * frame.uvStride + sourceColumn * frame.uvPixelStride
                    val targetIndex = row * uvWidth + column
                    rotatedU.put(targetIndex, frame.u.get(sourceIndex))
                    rotatedV.put(targetIndex, frame.v.get(sourceIndex))
                }
            }
            return YuvFrame(rotatedY, rotatedU, rotatedV, frame.width, frame.height, frame.width, uvWidth, 1)
        }

        fun copyFrame(image: Image): YuvFrame {
            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]
            val uvHeight = (image.height + 1) / 2

            return YuvFrame(
                y = copyPlane(yPlane, image.height),
                u = copyPlane(uPlane, uvHeight),
                v = copyPlane(vPlane, uvHeight),
                width = image.width,
                height = image.height,
                yStride = yPlane.rowStride,
                uvStride = uPlane.rowStride,
                uvPixelStride = uPlane.pixelStride
            )
        }

        private fun copyPlane(plane: Image.Plane, rowCount: Int): ByteBuffer {
            val rowStride = plane.rowStride
            val source = plane.buffer.duplicate()
            val destination = ByteBuffer.allocateDirect(rowStride * rowCount)
            val sourceStart = source.position()

            for (row in 0 until rowCount) {
                val sourceOffset = sourceStart + row * rowStride
                if (sourceOffset >= source.limit()) break
                source.position(sourceOffset)
                val bytesToCopy = minOf(rowStride, source.limit() - sourceOffset)
                destination.position(row * rowStride)
                destination.put(source.slice().apply { limit(bytesToCopy) })
            }

            destination.position(0)
            return destination
        }
    }
}

data class RgbYuvDetections(
    val rgb: List<Detection>,
    val yuv: List<Detection>
)

data class YuvFrame(
    val y: ByteBuffer,
    val u: ByteBuffer,
    val v: ByteBuffer,
    val width: Int,
    val height: Int,
    val yStride: Int,
    val uvStride: Int,
    val uvPixelStride: Int
)
