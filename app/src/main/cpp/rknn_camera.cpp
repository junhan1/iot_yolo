#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>
#include <rknn_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr char kTag[] = "RknnCamera";
constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.35F;
constexpr float kNmsThreshold = 0.45F;

struct Context { rknn_context handle{}; rknn_input_output_num io{}; };
struct Box { float left, top, right, bottom, score; int class_id; };

void logError(const char* message, int code) { __android_log_print(ANDROID_LOG_ERROR, kTag, "%s: %d", message, code); }

std::vector<uint8_t> readAsset(AAssetManager* manager, const char* name) {
  AAsset* asset = AAssetManager_open(manager, name, AASSET_MODE_BUFFER);
  if (!asset) return {};
  const auto size = static_cast<size_t>(AAsset_getLength(asset));
  std::vector<uint8_t> data(size);
  AAsset_read(asset, data.data(), size);
  AAsset_close(asset);
  return data;
}

std::vector<uint8_t> yuvToRgb(JNIEnv* env, jobject y_buffer, jobject u_buffer, jobject v_buffer, int width, int height, int y_stride, int uv_stride, int uv_pixel_stride) {
  auto* y = static_cast<uint8_t*>(env->GetDirectBufferAddress(y_buffer));
  auto* u = static_cast<uint8_t*>(env->GetDirectBufferAddress(u_buffer));
  auto* v = static_cast<uint8_t*>(env->GetDirectBufferAddress(v_buffer));
  if (!y || !u || !v || width <= 0 || height <= 0 || y_stride < width || uv_stride <= 0 || uv_pixel_stride <= 0) return {};

  const float scale = std::min(static_cast<float>(kInputSize) / width, static_cast<float>(kInputSize) / height);
  const int resized_width = std::max(1, static_cast<int>(width * scale));
  const int resized_height = std::max(1, static_cast<int>(height * scale));
  const int pad_left = (kInputSize - resized_width) / 2;
  const int pad_top = (kInputSize - resized_height) / 2;
  std::vector<uint8_t> rgb(kInputSize * kInputSize * 3, 114);

  const auto clamp = [](int value) { return static_cast<uint8_t>(std::clamp(value, 0, 255)); };
  for (int out_y = pad_top; out_y < pad_top + resized_height; ++out_y) {
    const int src_y = std::min(height - 1, static_cast<int>((out_y - pad_top) * height / static_cast<float>(resized_height)));
    for (int out_x = pad_left; out_x < pad_left + resized_width; ++out_x) {
      const int src_x = std::min(width - 1, static_cast<int>((out_x - pad_left) * width / static_cast<float>(resized_width)));
      const int yy = y[src_y * y_stride + src_x] - 16;
      const int uv_offset = (src_y / 2) * uv_stride + (src_x / 2) * uv_pixel_stride;
      const int uu = u[uv_offset] - 128;
      const int vv = v[uv_offset] - 128;
      const size_t index = (out_y * kInputSize + out_x) * 3;
      rgb[index] = clamp((298 * yy + 409 * vv + 128) >> 8);
      rgb[index + 1] = clamp((298 * yy - 100 * uu - 208 * vv + 128) >> 8);
      rgb[index + 2] = clamp((298 * yy + 516 * uu + 128) >> 8);
    }
  }
  return rgb;
}

float iou(const Box& a, const Box& b) {
  const float left = std::max(a.left, b.left), top = std::max(a.top, b.top);
  const float right = std::min(a.right, b.right), bottom = std::min(a.bottom, b.bottom);
  const float intersection = std::max(0.F, right - left) * std::max(0.F, bottom - top);
  const float union_area = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection;
  return union_area > 0 ? intersection / union_area : 0.F;
}

std::vector<Box> nms(std::vector<Box> boxes) {
  std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> selected;
  for (const auto& box : boxes) {
    const bool suppressed = std::any_of(selected.begin(), selected.end(), [&](const Box& kept) { return kept.class_id == box.class_id && iou(kept, box) >= kNmsThreshold; });
    if (!suppressed) selected.push_back(box);
  }
  return selected;
}

std::vector<Box> decodeOutput(const rknn_tensor_attr& attr, const rknn_output& output) {
  if (attr.n_dims != 3) return {};

  // RKNN 版本和导出布局不同，兼容 [1, C, N] 与 [1, N, C]。
  const int first = attr.dims[1];
  const int second = attr.dims[2];
  const bool channelsFirst = first >= 6 && first <= 256;
  const int channels = channelsFirst ? first : second;
  const int candidates = channelsFirst ? second : first;
  if (channels < 6 || candidates <= 0) return {};

  const auto* data = static_cast<const float*>(output.buf);
  auto valueAt = [&](int channel, int candidate) {
    return channelsFirst
        ? data[channel * candidates + candidate]
        : data[candidate * channels + channel];
  };

  std::vector<Box> boxes;
  boxes.reserve(candidates / 8);
  for (int candidate = 0; candidate < candidates; ++candidate) {
    int classId = 0;
    float score = 0.F;
    for (int klass = 4; klass < channels; ++klass) {
      const float value = valueAt(klass, candidate);
      if (value > score) {
        score = value;
        classId = klass - 4;
      }
    }
    if (!std::isfinite(score) || score < kConfidenceThreshold) continue;

    const float cx = valueAt(0, candidate);
    const float cy = valueAt(1, candidate);
    const float width = valueAt(2, candidate);
    const float height = valueAt(3, candidate);
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) ||
        !std::isfinite(height) || width <= 0.F || height <= 0.F) {
      continue;
    }
    boxes.push_back({
        (cx - width / 2.F) / kInputSize,
        (cy - height / 2.F) / kInputSize,
        (cx + width / 2.F) / kInputSize,
        (cy + height / 2.F) / kInputSize,
        score,
        classId});
  }
  return boxes;
}

}

extern "C" JNIEXPORT jlong JNICALL Java_com_example_rknncamera_RknnDetector_nativeCreate(JNIEnv* env, jobject, jobject assets, jstring asset_name) {
  const char* name = env->GetStringUTFChars(asset_name, nullptr);
  auto model = readAsset(AAssetManager_fromJava(env, assets), name);
  env->ReleaseStringUTFChars(asset_name, name);
  if (model.empty()) return 0;
  auto context = std::make_unique<Context>();
  int result = rknn_init(&context->handle, model.data(), model.size(), 0, nullptr);
  if (result != RKNN_SUCC) { logError("rknn_init failed", result); return 0; }
  result = rknn_query(context->handle, RKNN_QUERY_IN_OUT_NUM, &context->io, sizeof(context->io));
  if (result != RKNN_SUCC || context->io.n_input != 1) { logError("unsupported RKNN input", result); rknn_destroy(context->handle); return 0; }
  __android_log_print(ANDROID_LOG_INFO, kTag, "rknn_init success: inputs=%u outputs=%u model_bytes=%zu", context->io.n_input, context->io.n_output, model.size());
  return reinterpret_cast<jlong>(context.release());
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_example_rknncamera_RknnDetector_nativeDetect(JNIEnv* env, jobject, jlong value, jobject y, jobject u, jobject v, jint width, jint height, jint y_stride, jint uv_stride, jint uv_pixel_stride) {
  auto* context = reinterpret_cast<Context*>(value);
  auto rgb = yuvToRgb(env, y, u, v, width, height, y_stride, uv_stride, uv_pixel_stride);
  if (!context || rgb.empty()) return env->NewFloatArray(0);
  rknn_input input{};
  input.index = 0; input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC; input.size = rgb.size(); input.buf = rgb.data(); input.pass_through = 0;
  if (rknn_inputs_set(context->handle, 1, &input) != RKNN_SUCC || rknn_run(context->handle, nullptr) != RKNN_SUCC) { logError("rknn inference failed", -1); return env->NewFloatArray(0); }
  std::vector<rknn_output> outputs(context->io.n_output);
  for (uint32_t index = 0; index < context->io.n_output; ++index) { outputs[index].index = index; outputs[index].want_float = 1; }
  if (rknn_outputs_get(context->handle, context->io.n_output, outputs.data(), nullptr) != RKNN_SUCC) return env->NewFloatArray(0);

  rknn_tensor_attr attr{}; attr.index = 0;
  if (rknn_query(context->handle, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) == RKNN_SUCC) {
    const auto boxes = decodeOutput(attr, outputs[0]);
    __android_log_print(ANDROID_LOG_INFO, kTag, "output dims=%u,%u,%u candidates=%zu", attr.dims[0], attr.dims[1], attr.dims[2], boxes.size());
    rknn_outputs_release(context->handle, context->io.n_output, outputs.data());
    const auto final_boxes = nms(boxes);
    std::vector<float> packed; packed.reserve(final_boxes.size() * 6);
    for (const auto& box : final_boxes) packed.insert(packed.end(), {box.left, box.top, box.right, box.bottom, box.score, static_cast<float>(box.class_id)});
    auto result = env->NewFloatArray(packed.size());
    if (!packed.empty()) env->SetFloatArrayRegion(result, 0, packed.size(), packed.data());
    return result;
  }
  rknn_outputs_release(context->handle, context->io.n_output, outputs.data());
  return env->NewFloatArray(0);
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_example_rknncamera_RknnDetector_nativeDetectRgb(JNIEnv* env, jobject, jlong value, jobject rgb_buffer, jint width, jint height) {
  auto* context = reinterpret_cast<Context*>(value);
  auto* rgb = static_cast<const uint8_t*>(env->GetDirectBufferAddress(rgb_buffer));
  if (!context || !rgb || width != kInputSize || height != kInputSize) return env->NewFloatArray(0);

  rknn_input input{};
  input.index = 0; input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC;
  input.size = kInputSize * kInputSize * 3; input.buf = const_cast<uint8_t*>(rgb); input.pass_through = 0;
  if (rknn_inputs_set(context->handle, 1, &input) != RKNN_SUCC || rknn_run(context->handle, nullptr) != RKNN_SUCC) {
    logError("rknn RGB inference failed", -1);
    return env->NewFloatArray(0);
  }

  std::vector<rknn_output> outputs(context->io.n_output);
  for (uint32_t index = 0; index < context->io.n_output; ++index) { outputs[index].index = index; outputs[index].want_float = 1; }
  if (rknn_outputs_get(context->handle, context->io.n_output, outputs.data(), nullptr) != RKNN_SUCC) return env->NewFloatArray(0);

  rknn_tensor_attr attr{}; attr.index = 0;
  if (rknn_query(context->handle, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) == RKNN_SUCC) {
    const auto boxes = decodeOutput(attr, outputs[0]);
    __android_log_print(ANDROID_LOG_INFO, kTag, "output dims=%u,%u,%u candidates=%zu", attr.dims[0], attr.dims[1], attr.dims[2], boxes.size());
    rknn_outputs_release(context->handle, context->io.n_output, outputs.data());
    const auto final_boxes = nms(boxes);
    std::vector<float> packed; packed.reserve(final_boxes.size() * 6);
    for (const auto& box : final_boxes) packed.insert(packed.end(), {box.left, box.top, box.right, box.bottom, box.score, static_cast<float>(box.class_id)});
    auto result = env->NewFloatArray(packed.size());
    if (!packed.empty()) env->SetFloatArrayRegion(result, 0, packed.size(), packed.data());
    return result;
  }
  rknn_outputs_release(context->handle, context->io.n_output, outputs.data());
  return env->NewFloatArray(0);
}
extern "C" JNIEXPORT void JNICALL Java_com_example_rknncamera_RknnDetector_nativeDestroy(JNIEnv*, jobject, jlong value) {
  auto context = std::unique_ptr<Context>(reinterpret_cast<Context*>(value));
  if (context) rknn_destroy(context->handle);
}
