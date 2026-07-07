#pragma once

#include "rfdetr/core/types.hpp"

#include <cstddef>
#include <vector>

namespace rfdetr {

// Some RF-DETR exports include a no-object/background logit and some deployment
// exports do not. The engine sidecar controls which raw class id is filtered as
// background; -1 means all channels are foreground classes.
inline constexpr int kCocoBackgroundIndex = 0;

// Parameters for RF-DETR detection decoding.
struct PostprocessParams {
    int   num_queries{300};       // model output dim 1
    int   num_classes_with_bg{91};// model output dim 2
    int   topk{300};              // num_select; default = num_queries
    float threshold{0.5f};        // score threshold
    int   bg_class_index{kCocoBackgroundIndex};
};

// Map raw class id (0..num_classes_with_bg-1) to dense user-facing class id.
// Shifts down by one when background filtering is enabled, so callers never see
// the no-object slot.
inline int dense_class_from_raw(int raw_class, int bg_class_index) noexcept {
    return (bg_class_index >= 0) ? (raw_class - 1) : raw_class;
}

// Decode RF-DETR detection outputs (CPU implementation).
//
// Inputs (host pointers, contiguous):
//   dets   : (num_queries, 4) float32, cxcywh normalized to [0,1]
//   labels : (num_queries, num_classes_with_bg) float32, raw logits
//
// img_w / img_h: original-image dims for box rescaling.
//
// Output: dense Detections with xyxy boxes in original-image pixel coordinates.
Detections decode_detections(const float* dets, const float* labels, int img_w,
                              int img_h, const PostprocessParams& params);

// Same as decode_detections, but also returns the query index for each kept
// detection (parallel to the returned vector). Needed for segmentation mask
// gathering. Upstream RF-DETR uses `topk_boxes = topk_idx // C` to gather the
// corresponding per-query mask.
Detections decode_detections_with_queries(const float* dets, const float* labels,
                                           int img_w, int img_h,
                                           const PostprocessParams& params,
                                           std::vector<int>& out_query_idx);

// Decode RF-DETR segmentation masks for an existing list of detections.
//
// Inputs:
//   masks_logits  : (num_queries, mask_h, mask_w) float32, raw logits, host pointer.
//   query_indices : per-detection query index (parallel to `detections`).
//   mask_h/mask_w : mask resolution from the engine bindings.
//   img_w/img_h   : original image size.
//   stream        : when non-null, offloads bilinear upsample + threshold to the GPU.
//                   Pass nullptr to use the CPU fallback.
//
// Each detection's `.mask` field is populated with a CV_8UC1 binary mask of
// size (img_h, img_w), values 0 or 255. Matches the upstream Python behavior:
// bilinear upsample to full image, threshold at logit > 0 (sigmoid > 0.5),
// masks are NOT bbox-cropped.
void decode_masks(const float* masks_logits, int mask_h, int mask_w,
                  const std::vector<int>& query_indices,
                  int img_w, int img_h, Detections& detections,
                  void* stream = nullptr);

}  // namespace rfdetr
