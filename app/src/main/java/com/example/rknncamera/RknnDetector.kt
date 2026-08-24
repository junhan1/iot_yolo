package com.example.rknncamera

import android.content.res.AssetManager
import android.media.Image
import java.nio.ByteBuffer

class RknnDetector(assets: AssetManager, modelAssetName: String) : AutoCloseable {
    private var nativeHandle = nativeCreate(assets, modelAssetName)

    fun detect(frame: YuvFrame): List<Detection> {
        if (nativeHandle == 0L) return emptyList()
        return nativeDetect(
            nativeHandle,
            frame.y,
            frame.u,
            frame.v,
            frame.width,
            frame.height,
            frame.yStride,
            frame.uvStride,
            frame.uvPixelStride
        ).asList().chunked(6).mapNotNull { item ->
            item.takeIf { it.size == 6 }?.let {
                Detection(it[0], it[1], it[2], it[3], it[4], it[5].toInt())
            }
        }
    }

    override fun close() {
        if (nativeHandle != 0L) nativeDestroy(nativeHandle).also { nativeHandle = 0L }
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
    private external fun nativeDestroy(handle: Long)

    private fun FloatArray.asList() = toList()

    companion object {
        init { System.loadLibrary("rknn_camera") }

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