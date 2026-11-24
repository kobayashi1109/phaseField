// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1
//
// Laplace CPU/GPU Matrix-Free validation + NVTX profiling (direct GPU cells).
// NOTE: Reconstructed from the PRISMS-PF baseline and the described benchmark.
//       The uncommitted s73 main.cc was not accessible for a byte-for-byte edit.

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>
#include <prismspf/core/solve_block.h>

#ifdef PRISMS_PF_GPU
#  include "cpu_laplace_probe.h"
#  include <prismspf/solvers/portable_scalar_pde_operator.h>

#  include <deal.II/base/mpi.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/matrix_free/matrix_free.h>
#  include <deal.II/matrix_free/portable_matrix_free.h>

#  include <Kokkos_Core.hpp>
#  include <nvtx3/nvToolsExt.h>

#  include <algorithm>
#  include <chrono>
#  include <cmath>
#  include <iomanip>
#  include <iostream>
#  include <stdexcept>
#endif

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Preserve the original PRISMS-PF application setup.
  prismspf::MPIInitFinalize mpi_init(argc, argv);
  ParseCMDOptions cli_options(argc, argv);

  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 2;

  std::vector<FieldAttributes> fields = {FieldAttributes("u")};

  SolveBlock linear_solve;
  linear_solve.id               = 1;
  linear_solve.field_indices    = {0};
  linear_solve.solve_type       = Linear;
  linear_solve.solve_timing     = Uninitialized;
  linear_solve.dependencies_lhs = make_dependency_set(fields, {"grad(lhs(u))"});

  std::vector<SolveBlock> solve_blocks({linear_solve});

  UserInputParameters<dim>       user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields,
                                       solve_blocks,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  // CPU solver remains unchanged and is outside the profiled benchmark.
  problem.solve();

#ifdef PRISMS_PF_GPU
  // This probe assumes a single MPI process. For MPI>1, ghost exchange is required.
  if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) != 1)
    throw std::runtime_error("Laplace CPU/GPU probe requires one MPI process.");

  using HostVector = dealii::LinearAlgebra::distributed::Vector<double>;
  using DeviceVector =
    dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::Default>;

  const auto &matrix_free_manager =
    problem.get_solve_context().get_matrix_free_manager();

  const auto &cpu_mf = matrix_free_manager.get_shared_matrix_free();
  const auto &portable_mf =
    matrix_free_manager.get_portable_shared_matrix_free(0);

  HostVector src_cpu;
  HostVector dst_cpu;
  DeviceVector src_gpu;
  DeviceVector dst_gpu;

  cpu_mf.initialize_dof_vector(src_cpu, 0);
  cpu_mf.initialize_dof_vector(dst_cpu, 0);
  portable_mf.initialize_dof_vector(src_gpu);
  portable_mf.initialize_dof_vector(dst_gpu);

  // GPU operator with NO implicit distributed cell_loop() communication.
  using GpuLHS = prismspf::CustomPDE<dim, degree, double>::GpuLHS;
  prismspf::PortableScalarPDEOperator<dim, degree, double, GpuLHS>
    gpu_operator(matrix_free_manager, 0);
  std::cout << "[GPU path] Direct Kokkos cells (no cell_loop/compress)" << std::endl;

  // Confirm the partitioner assumptions of the direct single-GPU path.
  // This diagnostic runs once, outside the numerical validation and timers.
  const auto &portable_partitioner = portable_mf.get_vector_partitioner();
  std::cout << "[Check] MatrixFree partitioner exists: "
            << static_cast<bool>(portable_partitioner) << '\n';

  if (portable_partitioner)
    {
      std::cout << "[Check] MPI processes: "
                << portable_partitioner->n_mpi_processes() << '\n';
      std::cout << "[Check] Ghost DoFs: "
                << portable_partitioner->n_ghost_indices() << '\n';
      std::cout << "[Check] Import DoFs: "
                << portable_partitioner->n_import_indices() << '\n';
      std::cout << "[Check] Source partitioner matches: "
                << (src_gpu.get_partitioner().get() == portable_partitioner.get())
                << '\n';
      std::cout << "[Check] Destination partitioner matches: "
                << (dst_gpu.get_partitioner().get() == portable_partitioner.get())
                << std::endl;
    }

  const unsigned int n_owned = src_cpu.locally_owned_size();
  if (n_owned != src_gpu.locally_owned_size())
    throw std::runtime_error("Host/device owned DoF counts differ.");

  const auto &constrained_dofs = cpu_mf.get_constrained_dofs(0);

  // Deterministic non-uniform input; constrained DoFs are set to zero.
  for (unsigned int i = 0; i < n_owned; ++i)
    src_cpu.local_element(i) = static_cast<double>(i % 101) / 101.0;
  for (const auto i : constrained_dofs)
    src_cpu.local_element(i) = 0.0;

  // Initialize the GPU vector once, before any timing or NVTX capture.
  // The current single-rank probe uses locally identical CPU/GPU indexing.
  using DeviceMemory = dealii::MemorySpace::Default::kokkos_space;
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using DeviceView = Kokkos::View<double *, DeviceMemory, Unmanaged>;

  const unsigned int n_gpu_storage =
    src_gpu.locally_owned_size() +
    src_gpu.get_partitioner()->n_ghost_indices();

  Kokkos::View<double *, Kokkos::HostSpace> input_host(
    "laplace_probe_input", n_gpu_storage);
  for (unsigned int i = 0; i < n_gpu_storage; ++i)
    input_host(i) = (i < n_owned) ? src_cpu.local_element(i) : 0.0;

  DeviceView input_gpu(src_gpu.get_values(), n_gpu_storage);
  Kokkos::deep_copy(input_gpu, input_host);

  // --------------------------------------------------------------------------
  // One correctness check outside the timed benchmark.
  // --------------------------------------------------------------------------
  CpuLaplaceCell<dim, degree, double> cpu_operator;

  cpu_mf.cell_loop(&CpuLaplaceCell<dim, degree, double>::apply,
                   &cpu_operator,
                   dst_cpu,
                   src_cpu,
                   true);
  for (const auto i : constrained_dofs)
    dst_cpu.local_element(i) = src_cpu.local_element(i);

  dst_gpu = 0.0;
  gpu_operator.apply_cells(dst_gpu, src_gpu);
  portable_mf.copy_constrained_values(src_gpu, dst_gpu);
  Kokkos::fence();

  DeviceView result_gpu(dst_gpu.get_values(), n_gpu_storage);
  Kokkos::View<double *, Kokkos::HostSpace> result_host(
    "laplace_probe_result", n_gpu_storage);
  Kokkos::deep_copy(result_host, result_gpu);

  double error_sq = 0.0;
  double ref_sq   = 0.0;
  double gpu_sq   = 0.0;
  double max_abs  = 0.0;
  for (unsigned int i = 0; i < n_owned; ++i)
    {
      const double ref = dst_cpu.local_element(i);
      const double err = result_host(i) - ref;
      error_sq += err * err;
      ref_sq += ref * ref;
      gpu_sq += result_host(i) * result_host(i);
      max_abs = std::max(max_abs, std::abs(err));
    }

  std::cout << std::scientific << std::setprecision(12);
  std::cout << "[GPU probe] Portable MatrixFree DoFs: "
            << portable_mf.get_dof_handler().n_dofs() << '\n';
  std::cout << "[CPU/GPU compare] Constrained DoFs zeroed: "
            << constrained_dofs.size() << '\n';
  std::cout << "[GPU probe] Laplace result L2 norm: "
            << std::sqrt(gpu_sq) << '\n';
  std::cout << "[CPU/GPU compare] Relative L2 error: "
            << std::sqrt(error_sq / std::max(ref_sq, 1.0e-300)) << '\n';
  std::cout << "[CPU/GPU compare] Max abs error: " << max_abs << '\n';

  // --------------------------------------------------------------------------
  // Warm-up (excluded from BENCHMARK_500 NVTX capture).
  // --------------------------------------------------------------------------
  constexpr int n_warmup = 10;
  constexpr int n_repeat = 500;

  for (int i = 0; i < n_warmup; ++i)
    {
      cpu_mf.cell_loop(&CpuLaplaceCell<dim, degree, double>::apply,
                       &cpu_operator,
                       dst_cpu,
                       src_cpu,
                       true);
      for (const auto j : constrained_dofs)
        dst_cpu.local_element(j) = src_cpu.local_element(j);
    }

  for (int i = 0; i < n_warmup; ++i)
    {
      dst_gpu = 0.0;
      gpu_operator.apply_cells(dst_gpu, src_gpu);
      portable_mf.copy_constrained_values(src_gpu, dst_gpu);
      Kokkos::fence();
    }

  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::duration dt) {
    return std::chrono::duration<double, std::milli>(dt).count();
  };

  // This top-level NVTX capture includes ONLY the two 500-iteration loops.
  nvtxRangePushA("BENCHMARK_500");

  // --------------------------------------------------------------------------
  // CPU loop: one NVTX range for the CPU matrix-free cell_loop and one for
  // constrained DoFs. The cell_loop(..., true) includes destination zeroing.
  // --------------------------------------------------------------------------
  nvtxRangePushA("CPU_BENCH_500");
  const auto cpu_start = Clock::now();

  for (int i = 0; i < n_repeat; ++i)
    {
      nvtxRangePushA("CPU_MATVEC");
      cpu_mf.cell_loop(&CpuLaplaceCell<dim, degree, double>::apply,
                       &cpu_operator,
                       dst_cpu,
                       src_cpu,
                       true);
      nvtxRangePop();

      nvtxRangePushA("CPU_CONSTRAINTS");
      for (const auto j : constrained_dofs)
        dst_cpu.local_element(j) = src_cpu.local_element(j);
      nvtxRangePop();
    }

  const auto cpu_end = Clock::now();
  nvtxRangePop();  // CPU_BENCH_500

  // --------------------------------------------------------------------------
  // GPU loop: separate device zeroing, direct kernel, constrained DoFs, fence.
  // --------------------------------------------------------------------------
  Kokkos::fence();
  nvtxRangePushA("GPU_BENCH_500");
  const auto gpu_start = Clock::now();

  for (int i = 0; i < n_repeat; ++i)
    {
      nvtxRangePushA("GPU_ZERO");
      dst_gpu = 0.0;
      nvtxRangePop();

      // Explicit fence makes this NVTX range include GPU execution time.
      nvtxRangePushA("GPU_LOCAL_KERNEL_WAIT");
      gpu_operator.apply_cells(dst_gpu, src_gpu);
      Kokkos::fence();
      nvtxRangePop();

      nvtxRangePushA("GPU_CONSTRAINTS");
      portable_mf.copy_constrained_values(src_gpu, dst_gpu);
      nvtxRangePop();

      nvtxRangePushA("GPU_SYNC");
      Kokkos::fence();
      nvtxRangePop();
    }

  const auto gpu_end = Clock::now();
  nvtxRangePop();  // GPU_BENCH_500
  nvtxRangePop();  // BENCHMARK_500

  const double cpu_avg_ms = ms(cpu_end - cpu_start) / n_repeat;
  const double gpu_avg_ms = ms(gpu_end - gpu_start) / n_repeat;

  std::cout << "[Benchmark] DoFs: "
            << portable_mf.get_dof_handler().n_dofs() << '\n';
  std::cout << "[Benchmark] Repetitions: " << n_repeat << '\n';
  std::cout << "[Benchmark] CPU operator: " << cpu_avg_ms << " ms\n";
  std::cout << "[Benchmark] GPU operator: " << gpu_avg_ms << " ms\n";
  std::cout << "[Benchmark] Speedup: " << cpu_avg_ms / gpu_avg_ms << "x\n";
#endif

  return 0;
}
