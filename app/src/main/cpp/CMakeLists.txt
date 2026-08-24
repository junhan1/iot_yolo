cmake_minimum_required(VERSION 3.22.1)
project(rknn_camera)

add_library(rknn_camera SHARED rknn_camera.cpp)
find_library(log-lib log)
find_library(android-lib android)

set(RKNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION ${RKNN_ROOT}/librknn/${ANDROID_ABI}/librknnrt.so)

target_include_directories(rknn_camera PRIVATE ${RKNN_ROOT}/include)
target_link_libraries(rknn_camera rknnrt ${log-lib} ${android-lib})