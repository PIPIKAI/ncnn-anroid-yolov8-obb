cmake_minimum_required(VERSION 3.10)

project("yolov8-obb")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-2.4.13.7-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

set(libs_DIR ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${libs_DIR})
# 添加你的新 .so 库文件所在路径，这里假设位于 src/main/jniLibs/<abi> 目录下
include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(yolo SHARED src/yolo.cpp src/yolo.h)
target_link_libraries(yolo ncnn ${OpenCV_LIBS} )

#link_directories(${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI})

add_library(${CMAKE_PROJECT_NAME} SHARED
    # List C/C++ source files with relative paths to this CMakeLists.txt.
    native-lib.cpp
)


# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
    # List libraries link to the target library
    ncnn
    ${OpenCV_LIBS}
    android
    mediandk
    ${libs_DIR}/libyolo.so
    log)