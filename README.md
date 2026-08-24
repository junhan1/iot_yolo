# RKNN Camera Android Demo

Camera2 captures 640×640 YUV frames and passes them to RKNN through JNI. The supplied detection model has two classes: `no-vest` and `vest`.

## Required runtime dependency

This repository cannot redistribute Rockchip's RKNN Android Runtime. Obtain the runtime matching your RK3568 Android image from RKNN Toolkit2 and add the files listed in `app/src/main/cpp/README.md`.

## Build and run

1. Open `d:\project\iot_yolo` in Android Studio.
2. Use Gradle 8.7 from `gradle/wrapper/gradle-wrapper.properties`.
3. Install Android SDK Platform 35, NDK and CMake 3.31.5.
4. Add the RKNN Runtime headers and ABI libraries.
5. Connect an RK3568 Android device and run `app`.

The model file is `app/src/main/assets/helmet.sanitized-rk3568.rknn`. The native decoder expects standard non-end-to-end Ultralytics detection output `[1, 4 + class_count, candidate_count]`; it applies confidence filtering and NMS.