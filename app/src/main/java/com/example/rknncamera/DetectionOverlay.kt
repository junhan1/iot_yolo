package com.example.rknncamera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

data class Detection(val left: Float, val top: Float, val right: Float, val bottom: Float, val score: Float, val classId: Int)

class DetectionOverlay(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.GREEN; textSize = 32f }
    private var detections: List<Detection> = emptyList()

    fun setDetections(value: List<Detection>) { detections = value; invalidate() }

    override fun onDraw(canvas: Canvas) { //核心绘图函数
        super.onDraw(canvas)
        detections.forEach { item ->
            canvas.drawRect(item.left * width, item.top * height, item.right * width, item.bottom * height, boxPaint)
            val label = "${if (item.classId == 0) "no-vest" else "vest"} ${"%.0f".format(item.score * 100)}%"
            canvas.drawText(label, item.left * width, (item.top * height - 8).coerceAtLeast(textPaint.textSize), textPaint)
        }
    }
}