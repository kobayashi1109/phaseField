# Experimental GPU solvers for PRISMS-PF

This development snapshot extends PRISMS-PF's matrix-free finite-element implementation with a single-rank GPU linear solver (CG and Jacobi PCG) and GPU-resident explicit time stepping for the scalar Allen–Cahn example. The existing CPU solver paths are retained. This is experimental work, not a general GPU backend for all PRISMS-PF applications.

## Requirements

Reference environment used for the November 2026 benchmarks: Linux, NVIDIA V100S/V100 GPUs, CUDA Toolkit 12.0, GCC 11.4, Open MPI 4.1.1, CMake 3.31.12, Kokkos 4.2.01, and deal.II 9.6.0 built with CUDA-enabled Kokkos support. CMake 3.25 or later is required by the original project. PRISMS-PF requires a C++20-capable configuration; deal.II must have compatible MPI, LAPACK, zlib, p4est, and index-width settings. Use the same MPI and Kokkos installations for deal.II, PRISMS-PF, and the examples. Configure the Kokkos architecture for the actual GPU. The NVIDIA driver, compiler and third-party libraries must be installed separately. The existing build may fetch vendored dependencies.

## Build

After cloning the development branch, enter the repository root. Set paths for your machine, not the paths on the original benchmark machine:

```bash
export CUDA_HOME=/path/to/cuda
export KOKKOS_ROOT=/path/to/kokkos-install
export MPI_ROOT=/path/to/openmpi-install
export DEAL_II_DIR=/path/to/dealii-install/lib/cmake/deal.II
export PATH="$MPI_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_ROOT/lib:$KOKKOS_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
make -f Makefile.gpu JOBS=4
```

The helper uses the original CMake build system and builds the core library before the applications. Executables: `build/gpu/laplace/main` and `build/gpu/allen-cahn/main`. To use a different build directory, set `BUILD_DIR`. To pass additional options to the original CMake configuration, set `CORE_EXTRA_ARGS` or `APP_EXTRA_ARGS`. A new build directory is recommended when changing compilers or dependency installations.

## Run the benchmarks

The input files `benchmarks/gpu/laplace.prm` and `benchmarks/gpu/allen_cahn.prm` are to be copied from the validated reference runs when preparing the repository snapshot. Use single-rank execution. The reference Allen–Cahn input has mesh adaptivity disabled, zero-normal-gradient natural boundary conditions, 200 time steps and dt = 0.0005.

```bash
# From the repository root, after building and including the reference inputs:
python3 benchmarks/gpu/run_benchmarks.py laplace --refs 5 6 7 8 9 --repeats 3
python3 benchmarks/gpu/run_benchmarks.py allen-cahn --refs 7 8 9 10 --repeats 1
```

For a quick functional check, use `--refs 5 --repeats 1` for Laplace, or `--refs 7 --repeats 1` for Allen–Cahn. Select a visible CUDA device with `--device`; set CPU threads with `--threads`. Each invocation creates a new output directory under `build/gpu/runs`. Logs, input copies and VTU outputs are retained there.

| Mode | PRISMS_PF_D_GPU_CG | PRISMS_PF_D_JACOBI | PRISMS_PF_E_GPU_EXPLICIT |
|---|---:|---:|---:|
| Laplace CPU CG | 0 | 0 | 0 |
| Laplace GPU CG | 1 | 0 | 0 |
| Laplace CPU Jacobi PCG | 0 | 1 | 0 |
| Laplace GPU Jacobi PCG | 1 | 1 | 0 |
| Allen–Cahn CPU | 0 | 0 | 0 |
| Allen–Cahn GPU Resident | 0 | 0 | 2 |

`PRISMS_PF_C_GPU_VMULT=0` and `PRISMS_PF_C_GPU_VERIFY=0` disable the earlier bridge/probe flags during these solver comparisons. Compare `Linear Solver` wall time for Laplace and `Solve Blocks` wall time for Allen–Cahn. These are application timing scopes, not isolated kernel times. The script rejects a reported linear solver convergence failure. Inspect the GPU execution-space/mode messages in the logs.

## Validation and limitations

The single-field examples use the same numerical formulation as the CPU version. The Jacobi inverse diagonal and lumped inverse mass vector are prepared on the CPU; their application is on the GPU. The explicit solver retains its primary solution on the GPU and synchronizes it for CPU postprocessing and output. Single MPI rank only; fixed meshes; no AMR, multiple coupled primary fields, or multi-GPU support in the resident explicit path. The CPU comparison runs use the CPU execution path of the same GPU-enabled build.

The reference results were obtained up to 1,050,625 DoFs. Validate on a new installation before interpreting performance. Performance varies with GPU, CPU, dependency builds, mesh and runtime environment. Keep upstream copyright notices and license files intact.
