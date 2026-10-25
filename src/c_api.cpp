#include "rfdetr/c_api.h"

#include "rfdetr/core/log.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/tasks/segmenter.hpp"
#include "rfdetr/version.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Thread-local error storage
// ---------------------------------------------------------------------------

static thread_local std::string t_last_error;

static void set_error(const std::string& msg) noexcept {
    t_last_error = msg;
}

// ---------------------------------------------------------------------------
// Log callback — the C-ABI accepts a single string callback; forward it
// through the C++ library's typed log system so TRT messages, segmenter
// warnings, and graph-capture notes all funnel through the same handler.
// ---------------------------------------------------------------------------

static std::atomic<rfdetr_log_fn_t> g_c_log_callback{nullptr};

static void c_log_adapter(rfdetr::LogSeverity /*sev*/, const char* msg) noexcept {
    if (auto cb = g_c_log_callback.load(std::memory_order_acquire)) {
        cb(msg ? msg : "");
    }
}

// ---------------------------------------------------------------------------
// Opaque handle structs
// ---------------------------------------------------------------------------

struct rfdetr_detector_s {
    rfdetr::RFDetrDetector obj;
    explicit rfdetr_detector_s(rfdetr::RFDetrDetector&& d) : obj(std::move(d)) {}
};

struct rfdetr_segmenter_s {
    rfdetr::RFDetrSegmenter obj;
    explicit rfdetr_segmenter_s(rfdetr::RFDetrSegmenter&& s) : obj(std::move(s)) {}
};

struct SegmentationResult : rfdetr_segmentations_t {
    std::vector<rfdetr_segmentation_t> items;
    std::vector<cv::Mat> masks;

    explicit SegmentationResult(const rfdetr::Detections& input)
        : items(input.size()), masks(input.size()) {
        count = static_cast<int>(items.size());
        detections = items.data();
        for (std::size_t i = 0; i < input.size(); ++i) {
            const auto& d = input[i];
            auto& item = items[i];
            item.detection = {{d.box.x1, d.box.y1, d.box.x2, d.box.y2}, d.class_id, d.score};
            if (d.mask.empty()) continue;
            const cv::Rect roi = cv::boundingRect(d.mask);
            if (roi.empty()) continue;
            d.mask(roi).convertTo(masks[i], CV_8UC1, 1.0 / 255.0);
            item.x = roi.x;
            item.y = roi.y;
            item.width = roi.width;
            item.height = roi.height;
            item.step = static_cast<int>(masks[i].step);
            item.data = masks[i].data;
        }
    }
};

// ---------------------------------------------------------------------------
// Helper: convert rfdetr::Detections -> heap-allocated rfdetr_detections_t
// ---------------------------------------------------------------------------

static rfdetr_detections_t* pack_detections(const rfdetr::Detections& dets) {
    auto* out = new rfdetr_detections_t;
    out->count      = static_cast<int>(dets.size());
    out->detections = (out->count > 0)
                          ? new rfdetr_detection_t[static_cast<std::size_t>(out->count)]
                          : nullptr;

    for (int i = 0; i < out->count; ++i) {
        const auto& d  = dets[static_cast<std::size_t>(i)];
        auto&       od = out->detections[i];
        od.box.x1   = d.box.x1;
        od.box.y1   = d.box.y1;
        od.box.x2   = d.box.x2;
        od.box.y2   = d.box.y2;
        od.class_id = d.class_id;
        od.score    = d.score;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Helper: wrap raw BGR bytes as a cv::Mat without copying
// ---------------------------------------------------------------------------

static cv::Mat wrap_bgr(const uint8_t* bgr_data, int width, int height, int step) {
    if (width <= 0 || height <= 0 || step < width * 3) {
        throw std::invalid_argument(
            "rfdetr: wrap_bgr: invalid image dimensions (width=" + std::to_string(width) +
            " height=" + std::to_string(height) + " step=" + std::to_string(step) + ")");
    }
    return cv::Mat(height, width, CV_8UC3,
                   const_cast<uint8_t*>(bgr_data),
                   static_cast<std::size_t>(step));
}

// ---------------------------------------------------------------------------
// Public C API implementation
// ---------------------------------------------------------------------------

extern "C" {

const char* rfdetr_last_error(void) {
    return t_last_error.c_str();
}

void rfdetr_set_log_callback(rfdetr_log_fn_t callback) {
    g_c_log_callback.store(callback, std::memory_order_release);
    rfdetr::set_log_callback(callback ? &c_log_adapter : nullptr);
}

const char* rfdetr_version(void) {
    return rfdetr::version();
}

// ----- Detector -----

rfdetr_detector_t* rfdetr_detector_create(const char* engine_path, const char* meta_path) {
    t_last_error.clear();
    if (!engine_path) { set_error("rfdetr_detector_create: engine_path is NULL"); return nullptr; }
    try {
        rfdetr::RFDetrDetector det =
            meta_path
                ? rfdetr::RFDetrDetector(engine_path, meta_path)
                : rfdetr::RFDetrDetector(engine_path);
        return new rfdetr_detector_s(std::move(det));
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

void rfdetr_detector_destroy(rfdetr_detector_t* det) {
    delete det;
}

const char* rfdetr_detector_variant(const rfdetr_detector_t* det) {
    if (!det) return "";
    return det->obj.variant().c_str();
}
int rfdetr_detector_input_width(const rfdetr_detector_t* det) {
    return det ? det->obj.input_w() : 0;
}
int rfdetr_detector_input_height(const rfdetr_detector_t* det) {
    return det ? det->obj.input_h() : 0;
}
int rfdetr_detector_num_queries(const rfdetr_detector_t* det) {
    return det ? det->obj.num_queries() : 0;
}
int rfdetr_detector_num_classes(const rfdetr_detector_t* det) {
    return det ? det->obj.num_classes() : 0;
}

rfdetr_detections_t* rfdetr_detector_detect(rfdetr_detector_t* det,
                                             const uint8_t* bgr_data,
                                             int width, int height, int step,
                                             float threshold) {
    t_last_error.clear();
    if (!det)      { set_error("rfdetr_detector_detect: det is NULL");      return nullptr; }
    if (!bgr_data) { set_error("rfdetr_detector_detect: bgr_data is NULL"); return nullptr; }
    try {
        cv::Mat img = wrap_bgr(bgr_data, width, height, step);
        auto dets   = det->obj.detect(img, threshold);
        return pack_detections(dets);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

// ----- Memory management -----

void rfdetr_detections_free(rfdetr_detections_t* dets) {
    if (!dets) return;
    delete[] dets->detections;
    delete dets;
}

rfdetr_segmenter_t* rfdetr_segmenter_create(
    const char* engine_path, const char* meta_path, int use_cuda_graph) {
    t_last_error.clear();
    if (!engine_path) { set_error("rfdetr_segmenter_create: engine_path is NULL"); return nullptr; }
    try {
        rfdetr::SegmenterOptions options;
        options.use_cuda_graph = use_cuda_graph != 0;
        options.resize_antialias = true;
        auto seg = meta_path ? rfdetr::RFDetrSegmenter(engine_path, meta_path, options)
                             : rfdetr::RFDetrSegmenter(engine_path, options);
        return new rfdetr_segmenter_s(std::move(seg));
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("rfdetr_segmenter_create: unknown exception");
    }
    return nullptr;
}

void rfdetr_segmenter_destroy(rfdetr_segmenter_t* seg) { delete seg; }
int rfdetr_segmenter_input_width(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.input_w() : 0;
}
int rfdetr_segmenter_input_height(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.input_h() : 0;
}
int rfdetr_segmenter_cuda_graph_active(const rfdetr_segmenter_t* seg) {
    return seg && seg->obj.cuda_graph_active();
}

rfdetr_segmentations_t* rfdetr_segmenter_segment(
    rfdetr_segmenter_t* seg, const uint8_t* bgr_data,
    int width, int height, int step, float threshold) {
    t_last_error.clear();
    if (!seg || !bgr_data) {
        set_error("rfdetr_segmenter_segment: NULL segmenter or image");
        return nullptr;
    }
    try {
        auto detections = seg->obj.segment(wrap_bgr(bgr_data, width, height, step), threshold);
        return std::make_unique<SegmentationResult>(detections).release();
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("rfdetr_segmenter_segment: unknown exception");
    }
    return nullptr;
}

void rfdetr_segmentations_free(rfdetr_segmentations_t* result) {
    delete static_cast<SegmentationResult*>(result);
}

}  // extern "C"
