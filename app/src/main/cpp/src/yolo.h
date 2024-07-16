#ifndef YOLO_H
#define YOLO_H


#include <opencv2/core/core.hpp>

#include <net.h>
#include <iostream>

#ifdef YOLO_EXPORTS
#define YOLO_API __declspec(dllexport)
#else
#define YOLO_API __declspec(dllimport)
#endif


#if defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || defined(WIN64) || defined(_WIN64) || defined(_WIN64_)
#define PLATFORM_WINDOWS 1
#elif defined(ANDROID) || defined(_ANDROID_)
#define PLATFORM_ANDROID 1
#elif defined(__linux__)
#define PLATFORM_LINUX	 1
#elif defined(__APPLE__) || defined(TARGET_OS_IPHONE) || defined(TARGET_IPHONE_SIMULATOR) || defined(TARGET_OS_MAC)
#define PLATFORM_IOS	 1
#else
#define PLATFORM_UNKNOWN 1
#endif

const int MAX_STRIDE = 32;
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2

// 返回的结构体
struct Point {
    float x, y;
    Point(const float& px = 0, const float& py = 0) : x(px), y(py) {}
    Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
    Point& operator+=(const Point& p) {
        x += p.x;
        y += p.y;
        return *this;
    }
    Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
    Point operator*(const float coeff) const { return Point(x * coeff, y * coeff); }
};

struct RotatedBox {
    float x_ctr, y_ctr, w, h, a;
};

struct Object {
    cv::Rect_<float> rect;
    RotatedBox r_rect;
    int label;
    float prob;
};
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
class Yolo
{
public:
    Yolo();
    int class_num;

    // 加载模型
    int load(
#if defined(PLATFORM_ANDROID)
            AAssetManager* mgr,
#endif
            const char* model_path, float img_scale, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    // 检测
    int detect(const cv::Mat& img, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.4f);


private:
    ncnn::Net yolo;
    // 图片缩小的倍率
    float img_scale = 2.0;
    // 图片标准化均值和方差
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H
