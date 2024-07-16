#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "src/yolo.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static Yolo* g_yolo = 0;
static ncnn::Mutex lock;

cv::Mat bitmapToMat(JNIEnv *env, jobject bitmap) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    void *pixels = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    cv::Mat mat(info.height, info.width, CV_8UC4, pixels);

    cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);

    AndroidBitmap_unlockPixels(env, bitmap);

    return mat.clone();
}

extern "C"
{
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID rId;
static jfieldID labelId;
static jfieldID probId;



JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo;
        g_yolo = 0;
    }
}


JNIEXPORT jboolean JNICALL
Java_com_pipikai_github_YoloNcnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager, jint modelid,
                                         jint cpugpu) {
    if (modelid < 0 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] =
            {
                    "yolov8n-obb-sim-opt-fp16",
                    "yolov8s-obb-sim-opt-fp16",
            };

    const float mean_vals[3] ={103.53f, 116.28f, 123.675f};

    const float norm_vals[3] ={ 1 / 255.f, 1 / 255.f, 1 / 255.f };

    const char* modeltype = modeltypes[(int)modelid];

    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_yolo;
            g_yolo = 0;
        }
        else
        {
            if (!g_yolo)
                g_yolo = new Yolo;
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "load model %s", modeltype);

            g_yolo->load(mgr, modeltype, 2.0 , mean_vals, norm_vals, use_gpu);
            g_yolo->class_num = 15;

        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/pipikai/github/YoloNcnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "()V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    rId = env->GetFieldID(objCls, "r", "F");
    labelId = env->GetFieldID(objCls, "label", "I");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}
int imgWidth;
int imgHeight;

JNIEXPORT jobjectArray JNICALL
Java_com_pipikai_github_YoloNcnn_Detect(JNIEnv *env, jobject thiz, jobject bitmap) {

    cv::Mat img = bitmapToMat(env,bitmap);

    imgWidth = img.cols;
    imgHeight = img.rows;
    std::vector<Object> objects;

    g_yolo->detect(img,objects,0.4,0.4);
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
    for (size_t i = 0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId);

        env->SetFloatField(jObj, xId, objects[i].r_rect.x_ctr);
        env->SetFloatField(jObj, yId, objects[i].r_rect.y_ctr);
        env->SetFloatField(jObj, wId, objects[i].r_rect.w);
        env->SetFloatField(jObj, hId, objects[i].r_rect.h);
        env->SetFloatField(jObj, rId, objects[i].r_rect.a);
        env->SetIntField(jObj, labelId, objects[i].label);
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }


    return jObjArray;

}
}
