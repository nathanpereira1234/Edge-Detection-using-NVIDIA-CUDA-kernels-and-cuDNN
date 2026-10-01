# Edge Detection using NVIDIA CUDA kernels and cuDNN
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

<!-- Fill this in as you make real changes. Delete any line that isn't true. -->

- [ ] Rebuilt and ran the inherited baseline on my own machine (see hardware below)
- [ ] Profiled the baseline with NVIDIA Nsight Systems to find where per-frame time goes
- [ ] **Optimization:** _e.g. pinned (page-locked) host memory + `cudaMemcpyAsync` on a stream_
- [ ] **Optimization:** _e.g. host↔device copies moved inside the CUDA graph_
- [ ] _Any other fix, feature, or refactor, with a link to the commit_

---

## My results

**Hardware / software:** _GPU model · driver version · CUDA version · cuDNN version · OS_
**Test input:** _clip name, resolution, fps, duration_

| Version | GPU time per frame | Excl. I/O per frame | Total incl. I/O per frame |
|---|---|---|---|
| Inherited baseline (my machine) | _TBD_ | _TBD_ | _TBD_ |
| + _my optimization 1_ | _TBD_ | _TBD_ | _TBD_ |
| + _my optimization 2_ | _TBD_ | _TBD_ | _TBD_ |

**What I learned:** _2–4 sentences: what the profiler showed, what helped, what didn't._

---

## Build and run

**Tested on:** _fill in your OS / CUDA version once verified_ (upstream: Ubuntu 24.04, x86_64, CUDA 12.5)

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
- Possible further work: int8 or fully integer pipeline, batching multiple frames, overlapping CPU I/O with GPU work across threads.

## License

Respect the license of the upstream repository (https://github.com/alex-n-braun/coursera_cuda_at_scale) and keep the NVIDIA license headers in the helper files intact.
