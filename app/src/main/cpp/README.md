# RKNN Android Runtime

已放置 RK356X Android Runtime 文件，来源为 Rockchip/airockchip 的 `rknpu2` 仓库：

- `include/rknn_api.h`
- `include/rknn_matmul_api.h`
- `librknn/arm64-v8a/librknnrt.so`
- `librknn/armeabi-v7a/librknnrt.so`

下载来源：

- https://github.com/airockchip/rknpu2/tree/master/runtime/RK356X/Android/librknn_api

文件校验：

- `arm64-v8a/librknnrt.so`: 5,741,576 bytes，ELF64，AArch64
- `armeabi-v7a/librknnrt.so`: 3,682,784 bytes，ELF32，ARM
- `rknn_api.h`: 31,505 bytes
- `rknn_matmul_api.h`: 4,670 bytes

`CMakeLists.txt` 会根据 Android ABI 链接对应的 `librknnrt.so`。模型是 RK3568 目标模型，设备端还必须具备匹配的 RKNN NPU 内核驱动和固件。Rockchip 头文件带有专有许可声明，使用和再分发请遵循对应 SDK 许可。