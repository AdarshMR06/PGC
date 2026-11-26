# PGC Evaluation 2 – Theme 5: Occupancy and Block Size Tuning in CUDA

## Aim
Study how block size and dynamic shared-memory allocation affect theoretical occupancy and measured SAXPY kernel execution time on the detected NVIDIA GPU.

## Current verified environment
- GPU: NVIDIA GeForce RTX 5060 Ti
- Reported VRAM: 16 GB (16,311 MiB)
- Architecture: Blackwell
- Compute capability: 12.0
- CUDA Toolkit: 13.4
- NVIDIA driver: 610.88
- Windows; MSVC 14.51 detected

These values come from the user's uploaded `nvidia-smi` and compiler output. Re-run the program on the target computer to generate the remaining device-property values.

## Requirements
- NVIDIA CUDA Toolkit 13.4 or compatible toolkit supporting `sm_120`
- NVIDIA driver compatible with the installed toolkit
- Visual Studio C++ Build Tools (MSVC x64)
- Python 3 and Matplotlib for graph generation

## CUDA execution model
A grid contains thread blocks; each block contains threads. Threads execute in groups called warps (32 threads on this GPU's CUDA programming model). Streaming multiprocessors schedule resident blocks and warps. Occupancy is the ratio of active warps to the maximum number of warps supported per SM.

## Occupancy formula
`Theoretical occupancy (%) = (active warps per SM / maximum warps per SM) × 100`

The CUDA occupancy API estimates active blocks per SM subject to kernel resource use and device limits. This is a theoretical resource-based estimate, not a direct hardware utilization measurement.

## Experiments
The program sweeps block sizes 32, 64, 128, 256, 512 and 1024. It warms up each configuration, measures repeated launches with CUDA events, and reports the median timing to reduce sensitivity to occasional interruptions. The SAXPY result is checked against a CPU reference.

The shared-memory experiment compares 0, 4096 and 16384 bytes of dynamic shared memory per block. The extra memory is used to create controlled resource pressure, not to improve SAXPY's mathematical work. Therefore, timing changes should not be interpreted as a general benefit of shared-memory optimization.

## Build and run
Open the x64 Native Tools Command Prompt for Visual Studio, change to this repository directory, and run:

```cmd
scripts\build_and_run.bat
```

Or run manually:

```cmd
nvcc -O2 -arch=sm_120 --ptxas-options=-v src\occupancy_tuning.cu -o occupancy_tuning.exe
occupancy_tuning.exe
```

Optional arguments: `occupancy_tuning.exe <vector_elements> <repetitions>`. Default: 16,777,216 elements and 15 repetitions.

Generate graphs after a successful run:

```cmd
python -m pip install matplotlib
python scripts\plot_results.py
```

## Output files
- `results/gpu_properties.csv`
- `results/block_size_results.csv`
- `results/occupancy_results.csv`
- `results/shared_memory_results.csv`
- PNG graphs in `graphs/`

## Important interpretation notes
- The compiler's `ptxas` output is the source for registers per thread; do not replace its value with a guess.
- The occupancy CSV labels register usage as `compiler_reported_separately` until the actual compiler report is recorded.
- The occupancy calculation is based on CUDA occupancy APIs and the device's maximum warps per SM. It is not measured achieved occupancy.
- A SAXPY kernel is generally memory-bandwidth-sensitive; highest theoretical occupancy does not guarantee the fastest result.
- GPU clocks, temperature, background activity, power state and Windows display use can affect timings.
- No benchmark numbers are included in this repository until the program is run on the target GPU.

## Limitations
The dynamic shared-memory experiment deliberately consumes shared memory without contributing to SAXPY's useful computation. It isolates resource pressure but is not a production optimization. Windows WDDM and display activity may add timing variability.

## Industry application
Block-size and occupancy tuning is relevant to deep-learning operators, scientific simulations and HPC kernels, where register pressure, shared-memory demand and latency hiding influence throughput.

## Conclusion
The best block size must be selected from the actual measured results on the target GPU. Do not assume that 1024 threads or maximum theoretical occupancy is automatically optimal.
