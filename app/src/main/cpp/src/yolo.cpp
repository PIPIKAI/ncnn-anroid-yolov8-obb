#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}
static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static float IoU(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
    float intersectionArea = (a & b).area();
    float unionArea = a.area() + b.area() - intersectionArea;
    return intersectionArea / unionArea;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }

}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}
static float dot_2d(const Point& A, const Point& B) {
    return A.x * B.x + A.y * B.y;
}

static float cross_2d(const Point& A, const Point& B) {
    return A.x * B.y - B.x * A.y;
}

static void get_rotated_vertices(const RotatedBox& box, Point(&pts)[4]) {
    double theta = box.a;
    float cosTheta2 = (float)cos(theta) * 0.5f;
    float sinTheta2 = (float)sin(theta) * 0.5f;

    pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
    pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
    pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
    pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
    pts[2].x = 2 * box.x_ctr - pts[0].x;
    pts[2].y = 2 * box.y_ctr - pts[0].y;
    pts[3].x = 2 * box.x_ctr - pts[1].x;
    pts[3].y = 2 * box.y_ctr - pts[1].y;
}

static int get_intersection_points(const Point(&pts1)[4], const Point(&pts2)[4],
                                   Point(&intersections)[24])
{

    Point vec1[4], vec2[4];
    for (int i = 0; i < 4; i++) {
        vec1[i] = pts1[(i + 1) % 4] - pts1[i];
        vec2[i] = pts2[(i + 1) % 4] - pts2[i];
    }

    int num = 0;  // number of intersections
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float det = cross_2d(vec2[j], vec1[i]);
            if (fabs(det) <= 1e-14) {
                continue;
            }

            auto vec12 = pts2[j] - pts1[i];

            float t1 = cross_2d(vec2[j], vec12) / det;
            float t2 = cross_2d(vec1[i], vec12) / det;

            if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    {
        const auto& AB = vec2[0];
        const auto& DA = vec2[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++) {
            auto AP = pts1[i] - pts2[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts1[i];
            }
        }
    }

    {
        const auto& AB = vec1[0];
        const auto& DA = vec1[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++) {
            auto AP = pts2[i] - pts1[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts2[i];
            }
        }
    }

    return num;
}

static int convex_hull_graham(const Point(&p)[24], const int& num_in, Point(&q)[24])
{
    int t = 0;
    for (int i = 1; i < num_in; i++) {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
            t = i;
        }
    }
    auto& start = p[t];

    for (int i = 0; i < num_in; i++) {
        q[i] = p[i] - start;
    }

    auto tmp = q[0];
    q[0] = q[t];
    q[t] = tmp;

    float dist[24];
    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    std::sort(q + 1, q + num_in, [](const Point& A, const Point& B) -> bool {
        float temp = cross_2d(A, B);
        if (fabs(temp) < 1e-6) {
            return dot_2d(A, A) < dot_2d(B, B);
        }
        else {
            return temp > 0;
        }
    });

    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    int k;
    for (k = 1; k < num_in; k++) {
        if (dist[k] > 1e-8) {
            break;
        }
    }
    if (k == num_in) {
        q[0] = p[t];
        return 1;
    }
    q[1] = q[k];
    int m = 2;
    for (int i = k + 1; i < num_in; i++) {
        while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
            m--;
        }
        q[m++] = q[i];
    }

    return m;
}

static float polygon_area(const Point(&q)[24], const int& m) {
    if (m <= 2) {
        return 0;
    }

    float area = 0;
    for (int i = 1; i < m - 1; i++) {
        area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
    }

    return area / 2.0;
}

static float rotated_boxes_intersection(const RotatedBox& box1, const RotatedBox& box2) {
    Point intersectPts[24], orderedPts[24];

    Point pts1[4];
    Point pts2[4];
    get_rotated_vertices(box1, pts1);
    get_rotated_vertices(box2, pts2);

    int num = get_intersection_points(pts1, pts2, intersectPts);

    if (num <= 2) {
        return 0.0;
    }

    int num_convex = convex_hull_graham(intersectPts, num, orderedPts);
    return polygon_area(orderedPts, num_convex);
}



// 去除 Iou > 阈值的框
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].r_rect.w * faceobjects[i].r_rect.h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            //float inter_area = intersection_area(a, b);
            auto inter_area = rotated_boxes_intersection(a.r_rect, b.r_rect);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float IoU = inter_area / union_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
// 产生初始的监测框
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

// 构造softmax推理最终结果
static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 15;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = sigmoid(scores[k]);
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }

        if (score >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float angle = *(pred.row(i) + 4 * reg_max_1 + num_class);

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float cos = std::cos(angle);
            float sin = std::sin(angle);

            float xx = (pred_ltrb[2] - pred_ltrb[0]) * 0.5f;
            float yy = (pred_ltrb[3] - pred_ltrb[1]) * 0.5f;
            float xr = xx * cos - yy * sin;
            float yr = xx * sin + yy * cos;
            float xa = xr + pb_cx;
            float ya = yr + pb_cy;
            float wa = (pred_ltrb[2] + pred_ltrb[0]);
            float ha = (pred_ltrb[3] + pred_ltrb[1]);

            Object obj;
            obj.label = label;
            obj.prob = score;
            obj.r_rect.x_ctr = xa;
            obj.r_rect.y_ctr = ya;
            obj.r_rect.w = wa;
            obj.r_rect.h = ha;
            obj.r_rect.a = angle;
            objects.push_back(obj);
        }
    }
}



// 加载模型
int Yolo::load(
#if defined(PLATFORM_ANDROID)
        AAssetManager* mgr,
#endif
        const char* model_path, float _img_scale, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", model_path);
    sprintf(modelpath, "%s.bin", model_path);

#if defined(PLATFORM_ANDROID)
    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);
#elif defined(PLATFORM_WINDOWS)
    yolo.load_param(parampath);
    yolo.load_model(modelpath);
#endif
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    img_scale = _img_scale;
    return 0;
}


int Yolo::detect(const cv::Mat& img, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    const int MAX_STRIDES = 32;
    std::vector<int> strides = { 8, 16 ,32}; //
    int width = img.cols;
    int height = img.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    int target_w;
    int target_h;

    target_w = std::max(640.0f, int(width / 32) * 32 / img_scale);
    target_h = std::max(640.0f, int(height / 32) * 32 / img_scale);
    if (w > h)
    {
        scale = (float)target_w / w;
        w = target_w;
        h = h * scale;
    }
    else
    {
        scale = (float)target_h / h;
        h = target_h;
        w = w * scale;
    }


    ncnn::Extractor ex = yolo.create_extractor();


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
    // pad to target_size rectangle
    int wpad = (w + MAX_STRIDES - 1) / MAX_STRIDES * MAX_STRIDES - w;
    int hpad = (h + MAX_STRIDES - 1) / MAX_STRIDES * MAX_STRIDES - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    ex.input("images", in_pad);

    ncnn::Mat out;

    ex.extract("output0", out);
    //ex.extract("out", out);

    std::vector<GridAndStride> grid_strides;


    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    generate_proposals(grid_strides, out, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);

    double pi = M_PI;
    double pi_2 = M_PI_2;

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        float w_ = objects[i].r_rect.w > objects[i].r_rect.h ? objects[i].r_rect.w : objects[i].r_rect.h;
        float h_ = objects[i].r_rect.w > objects[i].r_rect.h ? objects[i].r_rect.h : objects[i].r_rect.w;
        float a_ = (float)std::fmod((objects[i].r_rect.w > objects[i].r_rect.h ? objects[i].r_rect.a : objects[i].r_rect.a + pi_2), pi);

        float xc = (objects[i].r_rect.x_ctr - (wpad / 2)) / scale;
        float yc = (objects[i].r_rect.y_ctr - (hpad / 2)) / scale;
        float w = w_ / scale;
        float h = h_ / scale;

        // clip
        xc = std::max(std::min(xc, (float)(width - 1)), 0.f);
        yc = std::max(std::min(yc, (float)(height - 1)), 0.f);
        w = std::max(std::min(w, (float)(width - 1)), 0.f);
        h = std::max(std::min(h, (float)(height - 1)), 0.f);


        objects[i].r_rect.x_ctr = xc;
        objects[i].r_rect.y_ctr = yc;
        objects[i].r_rect.w = w;
        objects[i].r_rect.h = h;
        objects[i].r_rect.a = a_;
    }

    return 0;
}

// 初始化
Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}
