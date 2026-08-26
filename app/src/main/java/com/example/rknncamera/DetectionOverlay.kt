package com.example.rknncamera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import java.util.Locale

data class Detection(val left: Float, val top: Float, val right: Float, val bottom: Float, val score: Float, val classId: Int)

class DetectionOverlay(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.GREEN; textSize = 32f }
    private val yuvBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val yuvTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.CYAN; textSize = 32f }
    private var detections: List<Detection> = emptyList()
    private var yuvDetections: List<Detection> = emptyList()
    private var comparisonMode = false
    private var drawCount = 0
    private var sourceWidth = 1f
    private var sourceHeight = 1f
    private var centerCrop = false

    fun setDetections(value: List<Detection>) {
        detections = value
        yuvDetections = emptyList()
        comparisonMode = false
        invalidate()
    }

    fun setComparisonDetections(rgb: List<Detection>, yuv: List<Detection>) {
        detections = rgb
        yuvDetections = yuv
        comparisonMode = true
        invalidate()
    }

    fun setRealtimeSourceSize(width: Int, height: Int) {
        sourceWidth = width.toFloat().coerceAtLeast(1f)
        sourceHeight = height.toFloat().coerceAtLeast(1f)
        centerCrop = true
        invalidate()
    }

    fun setImageSourceSize(width: Int, height: Int) {
        sourceWidth = width.toFloat().coerceAtLeast(1f)
        sourceHeight = height.toFloat().coerceAtLeast(1f)
        centerCrop = false
        invalidate()
    }

    fun setImageMode() {
        centerCrop = false
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val drawStartNs = System.nanoTime()
        super.onDraw(canvas)
        drawCount++
        drawDetections(canvas, detections, boxPaint, textPaint, if (comparisonMode) "RGB" else null)
        if (comparisonMode) {
            drawDetections(canvas, yuvDetections, yuvBoxPaint, yuvTextPaint, "YUV")
        }
        if (drawCount % 30 == 0) {
            val drawMs = (System.nanoTime() - drawStartNs) / 1_000_000.0
            Log.i("DetectionOverlay", "draw count=$drawCount rgb=${detections.size} yuv=${yuvDetections.size} drawMs=${"%.3f".format(Locale.US, drawMs)}")
        }
    }

    private fun drawDetections(
        canvas: Canvas,
        values: List<Detection>,
        boxPaint: Paint,
        textPaint: Paint,
        prefix: String?
    ) {
        values.forEach { item ->
            val label = (prefix?.let { "$it " } ?: "") + when (item.classId) {
                0 -> "head"
                1 -> "helmet"
                else -> "class-${item.classId}"
            } + " ${"%.0f".format(Locale.US, item.score.coerceIn(0f, 1f) * 100f)}%"
            val left = if (centerCrop) {
                item.left.coerceIn(0f, 1f) * sourceWidth * cropScale - cropLeft
            } else {
                item.left.coerceIn(0f, 1f) * sourceWidth * contentScaleX
            }
            val top = if (centerCrop) {
                item.top.coerceIn(0f, 1f) * sourceHeight * cropScale - cropTop
            } else {
                item.top.coerceIn(0f, 1f) * sourceHeight * contentScaleY
            }
            val right = if (centerCrop) {
                item.right.coerceIn(0f, 1f) * sourceWidth * cropScale - cropLeft
            } else {
                item.right.coerceIn(0f, 1f) * sourceWidth * contentScaleX
            }
            val bottom = if (centerCrop) {
                item.bottom.coerceIn(0f, 1f) * sourceHeight * cropScale - cropTop
            } else {
                item.bottom.coerceIn(0f, 1f) * sourceHeight * contentScaleY
            }
            canvas.drawRect(left, top, right, bottom, boxPaint)
            canvas.drawText(label, left, (top - 8).coerceAtLeast(textPaint.textSize), textPaint)
        }
    }

    private val cropScale: Float
        get() = if (centerCrop) maxOf(width / sourceWidth, height / sourceHeight) else 1f

    private val contentScaleX: Float
        get() = width / sourceWidth

    private val contentScaleY: Float
        get() = height / sourceHeight

    private val cropLeft: Float
        get() = if (centerCrop) (sourceWidth * cropScale - width) / 2f else 0f

    private val cropTop: Float
        get() = if (centerCrop) (sourceHeight * cropScale - height) / 2f else 0f
}
