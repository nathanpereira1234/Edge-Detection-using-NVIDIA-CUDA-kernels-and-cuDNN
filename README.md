# Edge Detection with CUDA Kernels and cuDNN — Reproduction & Extension

A GPU image/video edge-detection pipeline (Sobel-style edges, smoothed and in-painted back into the original frame) built from custom CUDA kernels, cuDNN convolutions, and a CUDA graph.

> **Attribution.** This repository builds on the work of **Alex N. Braun**, whose original project is the capstone of the Coursera *GPU Programming Specialization*:
> **Upstream:** https://github.com/alex-n-braun/coursera_cuda_at_scale
>
> The pipeline design, the CUDA/cuDNN implementation, and the optimization history described under [What I inherited](#what-i-inherited) are his. The NVIDIA helper headers (`helper_cuda.h`, `helper_string.h`) come from the CUDA Samples and keep their original license headers. My own contributions are listed separately under [What I changed](#what-i-changed) and [My results](#my-results).

---

## What I inherited

### Pipeline

Each image (or each video frame) passes through these stages on the GPU:

1. **uint8 → float** conversion (custom kernel `convertUint8ToFloat`).
2. **RGB → grayscale** as a 1×1 cuDNN convolution.
3. **Edge detection:** a cuDNN convolution with horizontal and vertical Sobel kernels, `pointwiseAbs`, then a convolution that merges the two edge channels into one.
4. **Edge thickening:** repeated smoothing convolutions, each followed by a `pointwiseMin` intensity clamp.
5. **Flat-region suppression:** a convolution that removes response in uniform areas, followed by a cutoff.
6. **Compositing:** the edge map is broadcast to RGBA, blended into the original image (`pointwiseHalo`), and the alpha channel is set (`setChannel`).
7. **float → uint8** conversion (`convertFloatToUint8`).

For video, the GPU stages are captured once into a **CUDA graph** and replayed for every frame.

### Code layout

| Path | Contents |
|---|---|
| `src/edgeDetection.cpp` | Entry point; single-image path and `processVideo` (graph-based) |
| `src/cuda_kernels.cu` | Custom element-wise kernels (conversions, abs/min, halo blend, channel set) |
| `include/filter.hpp` | `Filter::runFilterOnGpu`, the full pipeline |
| `include/convolution.hpp` | Wrapper around cuDNN convolution descriptors |
| `include/cuda_graph.hpp` | Capture and replay of the CUDA graph |
| `include/gpu_blob.hpp`, `src/gpu_blob.cu` | Device memory allocation and host↔device transfer |
| `include/gpu_session.hpp` | Long-lived cuDNN handle |
| `include/cli.hpp`, `include/io.hpp`, `src/io.cpp` | Command-line parsing; image (FreeImage) and video (OpenCV) I/O |
| `include/timer.hpp` | Timing utilities |

### Upstream optimization history (reported by the original author)

Upstream, the pipeline was optimized in steps, measured on a 10 s, 1280×720, 25 fps clip on the original author's hardware:

| Stage (upstream) | GPU time per frame | Total incl. I/O per frame |
|---|---|---|
| Naive implementation | ~33.6 ms | ~42.5 ms |
| Temp buffers reused instead of allocated per frame | ~31.6 ms | ~39.1 ms |
| cuDNN handle and descriptors created once, not per frame | ~6.0 ms | ~21.0 ms |
| Cached width/height setup | ~5.8 ms | ~19.7 ms |
| GPU stages captured in a CUDA graph | ~5.5 ms | ~19.0 ms |

These are **not my measurements**. My own numbers on my own hardware are in [My results](#my-results).

---

## What I changed

- [x] Rebuilt and ran the inherited baseline on my own setup (Google Colab, Tesla T4)
- [x] **Optimization:** page-locked (pinned) host memory for the CPU-side frame buffers. A custom STL allocator (`include/pinned_allocator.hpp`) backs `ImageCPU` with `cudaMallocHost` / `cudaFreeHost`, so host↔device copies DMA directly instead of going through the driver's staging buffer. Only `ImageCPU`'s storage type changed (`include/types.hpp`); the pipeline itself is untouched.
- [x] Verified correctness: the output for `data/Lena.png` is byte-identical before and after the change.

---

## My results

**Hardware / software:** Google Colab · Tesla T4 (driver 580.82.07) · CUDA 12.8 · cuDNN 9.8.0 · Ubuntu 24.04.1 LTS
**Test input:** ffmpeg `testsrc2` pattern, 1280×720, 25 fps, 10 s (250 frames)
**Method:** 3 runs per version, median of the per-frame times

| Version | GPU time per frame | Excl. I/O per frame | Total incl. I/O per frame |
|---|---|---|---|
| Inherited baseline (my machine) | 2.70 ms | 4.37 ms | 12.25 ms |
| + pinned host memory | 2.48 ms | 3.19 ms | 11.12 ms |

Host↔device transfer time per frame (excl. I/O minus GPU time) fell from **1.67 ms to 0.71 ms (−57%)**, and the non-I/O frame time fell by **27%**.

**What I learned:** Once the inherited optimizations (reused handles, CUDA graph) had brought GPU compute down to about 2.7 ms per frame, the two pageable-memory copies made up more than a third of the non-I/O frame time. Pinning the host buffers removed most of that overhead without touching a single kernel. The small drop in the GPU row (2.70 → 2.48 ms) is within the run-to-run variation on a shared Colab GPU, so I don't attribute it to this change. File I/O (OpenCV decode/encode) now dominates total time at about 8 ms per frame, so that is the next bottleneck.

---

## Build and run

**Tested on:** Google Colab, Ubuntu 24.04, Tesla T4, CUDA 12.8, cuDNN 9.8 (upstream: Ubuntu 24.04, x86_64, CUDA 12.5)

**Dependencies**

- CUDA Toolkit
- cuDNN (backend)
- FreeImage: `sudo apt install libfreeimage-dev`
- OpenCV: `sudo apt install libopencv-dev`

**Build**

```bash
make all
```

**Run on the sample image**

```bash
make run    # reads data/Lena.png, writes data/Lena_edge.png
```

**Custom input**

```bash
./bin/edgeDetection --input data/Lena.png --output data/Lena_edges.png
./bin/edgeDetection --input some_input.mp4 --output edges_video.mp4
```

**Clean**

```bash
make clean
```

---

## Known limitations

- A recorded CUDA graph is static, so the frame resolution must stay fixed for the whole video.
- Possible further work: overlapping CPU file I/O with GPU work (e.g. a decode thread plus double-buffered pinned frames and `cudaMemcpyAsync`), moving the copies into the CUDA graph, an integer-only pipeline, batching frames.

## License

Respect the license of the upstream repository (https://github.com/alex-n-braun/coursera_cuda_at_scale) and keep the NVIDIA license headers in the helper files intact.
