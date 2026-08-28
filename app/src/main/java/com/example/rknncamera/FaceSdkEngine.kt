package com.example.rknncamera

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.baidu.idl.main.facesdk.FaceAuth
import com.baidu.idl.main.facesdk.FaceDetect
import com.baidu.idl.main.facesdk.FaceFeature
import com.baidu.idl.main.facesdk.FaceInfo
import com.baidu.idl.main.facesdk.callback.Callback
import com.baidu.idl.main.facesdk.model.BDFaceImageInstance
import com.baidu.idl.main.facesdk.model.BDFaceInstance
import com.baidu.idl.main.facesdk.model.BDFaceSDKCommon
import com.baidu.idl.main.facesdk.model.BDFaceSDKConfig
import com.baidu.idl.main.facesdk.utils.PreferencesUtil
import java.util.concurrent.atomic.AtomicBoolean

/** 百度人脸 SDK 的最小封装：序列号激活、模型加载和图片特征提取。 */
class FaceSdkEngine {
    sealed interface State {
        data object Activating : State
        data object Loading : State
        data object Ready : State
        data class Failed(val code: Int, val message: String) : State
    }

    sealed interface FeatureResult {
        data class Success(val bytes: ByteArray) : FeatureResult
        data class Failed(val message: String) : FeatureResult
    }

    private val auth = FaceAuth()
    private lateinit var detector: FaceDetect
    private lateinit var feature: FaceFeature
    private val modelHandler = Handler(Looper.getMainLooper())
    private var modelLoadTimeout: Runnable? = null

    @Volatile
    private var ready = false
    private val modelLoading = AtomicBoolean(false)
    private val detectorModelReady = AtomicBoolean(false)
    private val featureModelReady = AtomicBoolean(false)
    private val modelLoadFailed = AtomicBoolean(false)
    private var detectorModelCallback: Callback? = null
    private var featureModelCallback: Callback? = null
    @Volatile
    private var modelStateListener: ((State) -> Unit)? = null

    fun activateAndInitialize(
        context: Context,
        serialNumber: String,
        onState: (State) -> Unit
    ) {
        val applicationContext = context.applicationContext
        PreferencesUtil.initPrefs(applicationContext)
        ready = false
        modelLoading.set(false)
        detectorModelReady.set(false)
        featureModelReady.set(false)
        modelLoadFailed.set(false)
        onState(State.Activating)
        auth.initLicenseOnLine(applicationContext, serialNumber, object : Callback {
            override fun onResponse(code: Int, response: String?) {
                Log.i(TAG, "在线激活回调：code=$code, response=${response.orEmpty()}")
                if (code != 0) {
                    onState(State.Failed(code, response.orEmpty()))
                    return
                }
                onState(State.Loading)
                startModelLoad(applicationContext, onState)
            }
        })
    }

    /**
     * 在线激活成功但 SDK 回调未及时返回时，使用 SDK 已落盘的授权状态继续加载模型。
     */
    fun loadActivatedLicenseModels(context: Context, onState: (State) -> Unit) {
        startModelLoad(context.applicationContext, onState)
    }

    fun extract(bitmap: Bitmap): FeatureResult {
        if (!ready) return FeatureResult.Failed("人脸 SDK 尚未就绪")
        val image = BDFaceImageInstance(bitmap)
        return try {
            val faces: Array<FaceInfo>? = detector.detect(
                BDFaceSDKCommon.DetectType.DETECT_VIS,
                image
            )
            val face = faces?.firstOrNull()
                ?: return FeatureResult.Failed("未检测到清晰人脸，请更换正脸照片")
            val bytes = ByteArray(FEATURE_SIZE)
            val size = feature.feature(
                BDFaceSDKCommon.FeatureType.BDFACE_FEATURE_TYPE_LIVE_PHOTO,
                image,
                face.landmarks,
                bytes
            )
            if (size > 0f) FeatureResult.Success(bytes)
            else FeatureResult.Failed("特征提取失败，SDK 返回 $size")
        } catch (error: Throwable) {
            FeatureResult.Failed("特征提取异常：${error.message ?: error.javaClass.simpleName}")
        } finally {
            image.destory()
        }
    }

    fun similarity(first: ByteArray, second: ByteArray): Float = feature.featureCompare(
        BDFaceSDKCommon.FeatureType.BDFACE_FEATURE_TYPE_LIVE_PHOTO,
        first,
        second,
        true
    )

    private fun startModelLoad(context: Context, onState: (State) -> Unit) {
        modelStateListener = onState
        if (ready) {
            Log.i(TAG, "模型已就绪，复用当前实例")
            onState(State.Ready)
            return
        }
        if (modelLoading.compareAndSet(false, true)) {
            val missingModel = REQUIRED_MODELS.firstOrNull { model ->
                runCatching { context.assets.open(model).close() }.isFailure
            }
            if (missingModel != null) {
                modelLoading.set(false)
                onState(State.Failed(MODEL_ASSET_MISSING_CODE, "缺少模型资源：$missingModel"))
                return
            }
            Log.i(TAG, "开始加载人脸检测和特征模型")
            onState(State.Loading)
            modelLoadTimeout?.let(modelHandler::removeCallbacks)
            modelLoadTimeout = Runnable {
                if (modelLoading.compareAndSet(true, false)) {
                    modelLoadFailed.set(true)
                    detectorModelCallback = null
                    featureModelCallback = null
                    Log.e(TAG, "人脸模型加载超时")
                    modelStateListener?.invoke(State.Failed(MODEL_LOAD_TIMEOUT_CODE, "模型加载超时，未收到 SDK 回调"))
                }
            }.also { modelHandler.postDelayed(it, MODEL_LOAD_TIMEOUT_MS) }
            loadModels(context)
        } else {
            Log.i(TAG, "模型正在加载，等待当前请求完成")
        }
    }

    private fun loadModels(context: Context) {
        detectorModelReady.set(false)
        featureModelReady.set(false)
        modelLoadFailed.set(false)
        val instance = BDFaceInstance().apply { creatInstance() }
        detector = FaceDetect(instance)
        feature = FaceFeature()
        val config = BDFaceSDKConfig().apply {
            maxDetectNum = 1
            minFaceSize = 80
        }
        detector.loadConfig(config)
        detectorModelCallback = createDetectorModelCallback()
        featureModelCallback = createFeatureModelCallback()
        detector.initModel(
            context,
            DETECT_MODEL,
            ALIGN_MODEL,
            BDFaceSDKCommon.DetectType.DETECT_VIS,
            BDFaceSDKCommon.AlignType.BDFACE_ALIGN_TYPE_RGB_ACCURATE,
            detectorModelCallback!!
        )
        feature.initModel(context, ID_FEATURE_MODEL, LIVE_FEATURE_MODEL, "", featureModelCallback!!)
    }

    private fun createDetectorModelCallback() = object : Callback {
        override fun onResponse(code: Int, response: String?) {
            Log.i(TAG, "检测模型加载回调：code=$code, response=${response.orEmpty()}")
            if (code != 0) {
                reportModelFailure(code, response)
                return
            }
            detectorModelReady.set(true)
            reportReadyWhenModelsLoaded()
        }
    }

    private fun createFeatureModelCallback() = object : Callback {
        override fun onResponse(code: Int, response: String?) {
            Log.i(TAG, "特征模型加载回调：code=$code, response=${response.orEmpty()}")
            if (code != 0) {
                reportModelFailure(code, response)
                return
            }
            featureModelReady.set(true)
            reportReadyWhenModelsLoaded()
        }
    }

    private fun reportReadyWhenModelsLoaded() {
        // FaceDetect 的回调仅用于上报加载失败；官方示例以特征模型回调作为初始化完成信号。
        if (featureModelReady.get() && !modelLoadFailed.get()) {
            ready = true
            modelLoading.set(false)
            modelLoadTimeout?.let(modelHandler::removeCallbacks)
            detectorModelCallback = null
            featureModelCallback = null
            Log.i(TAG, "全部人脸模型加载完成")
            modelStateListener?.invoke(State.Ready)
        }
    }

    private fun reportModelFailure(code: Int, response: String?) {
        if (modelLoadFailed.compareAndSet(false, true)) {
            ready = false
            modelLoading.set(false)
            modelLoadTimeout?.let(modelHandler::removeCallbacks)
            detectorModelCallback = null
            featureModelCallback = null
            modelStateListener?.invoke(State.Failed(code, response.orEmpty()))
        }
    }

    private companion object {
        const val TAG = "FaceSdkEngine"
        const val FEATURE_SIZE = 512
        const val MODEL_ASSET_MISSING_CODE = -1003
        const val MODEL_LOAD_TIMEOUT_CODE = -1002
        const val MODEL_LOAD_TIMEOUT_MS = 15_000L
        const val DETECT_MODEL = "face-sdk-models/detect/detect_rgb-customized-pa-faceid6_0.model.int8-0.0.11.1"
        const val ALIGN_MODEL = "face-sdk-models/align/align_rgb-customized-pa-model.model.float32-6.4.10.1"
        const val ID_FEATURE_MODEL = "face-sdk-models/feature/feature_id-mnasnet-pa-renzheng.model.int8-2.0.135.2"
        const val LIVE_FEATURE_MODEL = "face-sdk-models/feature/feature_live-mnasnet-pa-life.model.int8-2.0.134.1"
        val REQUIRED_MODELS = listOf(DETECT_MODEL, ALIGN_MODEL, ID_FEATURE_MODEL, LIVE_FEATURE_MODEL)
    }
}