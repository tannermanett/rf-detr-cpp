# GPU memory budget

Set `RFDETR_GPU_MEM_LIMIT` before the consumer process loads an engine:

```sh
export RFDETR_GPU_MEM_LIMIT=2G
```

PowerShell: `$env:RFDETR_GPU_MEM_LIMIT = "2G"`. For Docker, pass
`-e RFDETR_GPU_MEM_LIMIT=2G` to the worker container. Start with 2 GiB per process
as a trial setting: short local 720p/1080p inference checks peaked at 351 MiB
accounted for the player engine and 239 MiB for the jersey engine. Validate a
full production L4 job before adopting it as a default. Rebuild the native
library to use this feature; older binaries ignore the variable.

Accepted values are unsigned integer bytes, or binary K/M/G units: `512M`,
`6GB`, `6GiB`, `1024K`, `100B` (case-insensitive). Surrounding whitespace is
accepted. Unset, empty, and `0` mean unlimited. Decimals, signs, unknown suffixes,
and overflow are errors. The environment is read once per process. Malformed
configuration remains an error on every subsequent detector-creation attempt.

Over-budget allocations fail before `cudaMalloc`, with the caller, requested
bytes, current usage, ceiling, and remaining bytes in the exception. The existing
C ABI catches it; `rfdetr_last_error()` and the Python ctypes predictor preserve
its message. No C ABI changes are needed.

## Scope and limitations

This caps **accounted allocations**, not total process VRAM. All RF-DETR instances
using the same library in one process share the counter. Separate processes have
independent counters, even if they inherit the same environment value.

| Accounted allocation | Size source |
| --- | --- |
| TensorRT engine weights | Serialized plan size, an estimate |
| Context enqueue scratch | `getDeviceMemorySizeV2()` (legacy API on TRT 10.0) |
| Input/output device buffers | Requested allocation bytes |
| Preprocessing source image | Requested allocation bytes |
| Mask decode logits, indices, masks | Requested allocation bytes |

Unaccounted usage includes CUDA context/module overhead, driver rounding,
TensorRT persistent/internal or plugin allocations beyond the scratch query,
CUDA Graph resources, and allocations from PyTorch, ONNX Runtime, and SAM 2.
Pinned host memory is system RAM and is excluded. Plan size can overestimate or
underestimate weights, so the counter is not a strict lower bound either.
Global GPU OOM can still occur below this cap.

NVIDIA documents the enqueue-memory query and allocator interception in
[How TensorRT works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html).
A custom `IGpuAllocator` would improve TensorRT accounting, while still requiring
headroom for CUDA and allocation paths outside that allocator.

Statsyuk's Pass 1 detection, enrichment worker, and Pass 2 use separate processes.
This variable only budgets native RF-DETR allocations inside each consumer;
it does not partition VRAM among stages. Measure all concurrent consumers on the
target GPU before choosing limits. This change imposes no production default.

## Ownership and diagnostics

`dev_alloc()` reserves before allocation and rolls back on CUDA failure. Its
`DevPtr` deleter returns the charge after successful free, including during
exception unwinding. Failed frees retain the charge and log the error.
`host_alloc()` owns pinned memory immediately without charging the GPU budget.
New runtime allocations must use these helpers.

When growing a buffer, reset its old pointer and capacity before allocating a
replacement. Hold new device/host allocations in local RAII pointers until both
succeed. Dynamic-shape allocation failures invalidate shape setup, so retrying
the same shape cannot skip missing buffers. Engine reservations outlive TRT
objects on both normal teardown and failed construction.

`rfdetr::set_log_severity(rfdetr::LogSeverity::kInfo)` enables engine-load
accounting logs, describing weights and context before binding allocation.
Internal `gpu_budget::in_use()` and `peak()` report accounted reservations;
peak includes reservations whose CUDA allocation subsequently failed.
These internal functions are not an installed public API.

## Validation

CPU tests can be configured without CUDA, TensorRT, or OpenCV:

```sh
cmake -S tests -B /var/tmp/rfdetr-budget-tests
cmake --build /var/tmp/rfdetr-budget-tests --config Release
ctest --test-dir /var/tmp/rfdetr-budget-tests -C Release --output-on-failure
```

Tests cover parsing, persistent environment errors, exact-limit rejection,
overflow, peaks, concurrency, and RAII moves/unwinding. A CUDA test double covers
the real allocation wrappers, injected allocation/free failures, pinned-host
exclusion, and rejection before touching CUDA. It does not validate the driver.

For native validation, use C++17, CUDA 12+, TensorRT 10/11, and OpenCV 4.5+.
Select the target GPU architecture explicitly (89 for L4/RTX 40xx):

```sh
cmake -S . -B /var/tmp/rfdetr-native-tests -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_C_API=ON
cmake --build /var/tmp/rfdetr-native-tests -j2
ctest --test-dir /var/tmp/rfdetr-native-tests --output-on-failure
```

Before enabling a production cap, test normal inference and repeated rejected
construction with the actual production engine/runtime, then compare accounted
peaks with observed per-process VRAM under concurrent load.

Local verification on 2026-09-10: both CPU tests passed on MSVC 19.44 and GCC
13.3. The full native library, C ABI, apps, and examples built with CUDA 12.8,
TensorRT 10.9, and OpenCV 4.6. RTX 4060 GPU checks passed for the existing player
and jersey engines, including player inference with a 6 GiB cap. Additional
native checks passed for repeated malformed/over-budget C ABI construction,
preprocessing growth rejection followed by a smaller-frame retry, and repeated
dynamic-shape rejection followed by smaller-batch inference on a synthetic
engine. Counters returned to zero after teardown. Production L4 concurrency,
TensorRT 10.8/11, and a full segmentation engine were not validated in this run.
