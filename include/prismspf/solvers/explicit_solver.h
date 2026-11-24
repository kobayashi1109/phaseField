// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/core/types.h>

#include <prismspf/solvers/mf_operator.h>
#include <prismspf/solvers/solver_base.h>

#include <prismspf/utilities/timer.h>

#include <prismspf/config.h>

#ifdef PRISMS_PF_GPU
#  include <deal.II/base/memory_space.h>
#  include <deal.II/base/numbers.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/matrix_free/portable_matrix_free.h>
#  include <deal.II/matrix_free/portable_matrix_free.templates.h>
#  include <Kokkos_Core.hpp>

#  include <cstddef>
#  include <cstdlib>
#  include <cstring>
#  include <iostream>
#  include <limits>
#  include <memory>
#  include <stdexcept>
#endif

PRISMS_PF_BEGIN_NAMESPACE

#ifdef PRISMS_PF_GPU
namespace ExperimentalGPU
{
  // 0: CPU, 1: original bridge, 2: opt-in GPU-resident explicit stepping.
  inline int
  explicit_mode()
  {
    const char *env = std::getenv("PRISMS_PF_E_GPU_EXPLICIT");
    if (env == nullptr || std::strcmp(env, "0") == 0)
      return 0;
    if (std::strcmp(env, "1") == 0)
      return 1;
    if (std::strcmp(env, "2") == 0)
      return 2;
    throw std::runtime_error("PRISMS_PF_E_GPU_EXPLICIT must be 0, 1 or 2");
  }

  // A free function keeps the CUDA kernel type independent of private solver
  // members. The pointers reference device memory only.
  template <typename Number>
  void
  explicit_scale_inverse_mass(Number *dst, const Number *invm, const int n)
  {
    using ExecutionSpace =
      typename dealii::MemorySpace::Default::kokkos_space::execution_space;
    Kokkos::parallel_for(
      "prismspf::explicit_inverse_mass_E1",
      Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const int i) { dst[i] *= invm[i]; });
  }
} // namespace ExperimentalGPU
#endif

template <unsigned int dim, unsigned int degree, typename number>
class SolveContext;

/**
 * @brief This class handles the explicit solves of all explicit fields
 */
template <unsigned int dim, unsigned int degree, typename number>
class ExplicitSolver : public SolverBase<dim, degree, number>
{
  using SolverBase<dim, degree, number>::solutions;
  using SolverBase<dim, degree, number>::solve_context;
  using SolverBase<dim, degree, number>::solve_block;

public:
  /**
   * @brief Constructor.
   * @pre Solve context has initialized members.
   */
  ExplicitSolver(SolveBlock                               _solve_block,
                 const SolveContext<dim, degree, number> &_solve_context)
    : SolverBase<dim, degree, number>(_solve_block, _solve_context)
  {}

  void
  init(const std::list<SolveBlock> &all_solve_blocks) override
  {
#ifdef PRISMS_PF_GPU
    gpu_state.reset();
#endif
    SolverBase<dim, degree, number>::init(all_solve_blocks);
    // Initialize rhs_operator
    rhs_operator.init(solve_context->get_pde_operator(),
                      &PDEOperatorBase<dim, degree, number>::compute_rhs,
                      solve_context->get_field_attributes(),
                      solve_context->get_solution_indexer(),
                      solve_context->get_matrix_free_manager(),
                      solve_context->get_simulation_timer(),
                      solve_block,
                      solve_block.dependencies_rhs);
    rhs_operator.set_scaling_diagonal(
      true,
      solve_context->get_invm_manager().get_invm(solve_context->get_field_attributes(),
                                                 solve_block.field_indices));
  }

#ifdef PRISMS_PF_GPU
  void
  reinit() override
  {
    gpu_state.reset();
    SolverBase<dim, degree, number>::reinit();
  }
#endif

  /**
   * @brief Solve for a single update step.
   */
  void
  solve_impl() override
  {
    Timer::Scope scope("Explicit Solver");

    // Zero out the ghosts
    solutions.zero_out_ghosts();

#ifdef PRISMS_PF_GPU
    if (!try_gpu_explicit_step())
#endif
      rhs_operator.compute_operator(solutions.get_solution_full_vector());

    // Apply constraints
    solutions.apply_constraints();

    // Update the ghosts
    solutions.update_ghosts();
  }

#ifdef PRISMS_PF_GPU
  // Resident-mode host access is coordinated by Problem::solve_increment.
  // The latest device result is in old_solution after each device swap.
  // These functions are called before the normal end-of-increment host swap.
  [[nodiscard]] bool
  has_resident_solution() const
  {
    return gpu_state && gpu_state->resident && gpu_state->has_result;
  }

  void
  sync_resident_current_to_host()
  {
    if (!has_resident_solution() || gpu_state->host_current_valid)
      return;

    Timer::Scope copy_scope("E1 D2H");
    const unsigned int field_index = *solve_block.field_indices.begin();
    auto &host_current = solutions.get_solution_vector(field_index);
    const std::size_t n = gpu_state->old_solution.locally_owned_size();
    Kokkos::deep_copy(HostWrite(host_current.get_values(), n),
                      DeviceRead(gpu_state->old_solution.get_values(), n));
    Kokkos::fence();
    gpu_state->host_current_valid = true;
  }

  // At the final increment (including an early stop), restore the previous
  // host state too, so the subsequent normal host history swap preserves
  // the original Problem API semantics for Current and OldOne.
  void
  sync_resident_previous_to_host()
  {
    if (!has_resident_solution())
      return;

    Timer::Scope copy_scope("E1 Final Previous D2H");
    const unsigned int field_index = *solve_block.field_indices.begin();
    auto &host_previous = solutions.get_old_solution_vector(0, field_index);
    const std::size_t n = gpu_state->new_solution.locally_owned_size();
    Kokkos::deep_copy(HostWrite(host_previous.get_values(), n),
                      DeviceRead(gpu_state->new_solution.get_values(), n));
    Kokkos::fence();
  }

  // Preserve the existing Problem finite-value check (sample of entry 0),
  // without transferring the entire field on non-output increments.
  [[nodiscard]] bool
  resident_sample_not_finite() const
  {
    if (!has_resident_solution())
      throw std::runtime_error("E1 resident sample requested before first GPU step");

    if (gpu_state->host_current_valid)
      {
        const auto &host_current =
          solutions.get_solution_vector(*solve_block.field_indices.begin());
        return !dealii::numbers::is_finite(host_current.local_element(0));
      }

    Timer::Scope copy_scope("E1 NaN Scalar D2H");
    number value = number(0);
    Kokkos::deep_copy(HostWrite(&value, 1),
                      DeviceRead(gpu_state->old_solution.get_values(), 1));
    Kokkos::fence();
    return !dealii::numbers::is_finite(value);
  }

  // CPU labels still rotate at the end of each increment. A valid Current
  // mirror is therefore no longer a valid Current mirror after update().
  void
  update() override
  {
    SolverBase<dim, degree, number>::update();
    if (has_resident_solution())
      gpu_state->host_current_valid = false;
  }
#endif

private:
#ifdef PRISMS_PF_GPU
  // Bridge mode: CPU Current/OldOne remain authoritative.
  // Resident mode: device history is authoritative until an explicit sync.
  // Both modes cache device allocations and the CPU-computed inverse mass.
  using DeviceVector = typename PDEOperatorBase<dim, degree, number>::DeviceVector;
  using DeviceSpace = dealii::MemorySpace::Default::kokkos_space;
  using ExecutionSpace = typename DeviceSpace::execution_space;
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using DeviceWrite = Kokkos::View<number *, DeviceSpace, Unmanaged>;
  using DeviceRead = Kokkos::View<const number *, DeviceSpace, Unmanaged>;
  using HostWrite = Kokkos::View<number *, Kokkos::HostSpace, Unmanaged>;
  using HostRead = Kokkos::View<const number *, Kokkos::HostSpace, Unmanaged>;

  struct GPUExplicitState
  {
    DeviceVector old_solution;
    DeviceVector new_solution;
    DeviceVector inverse_mass;
    bool resident = false;
    bool has_result = false;
    bool host_current_valid = false;
  };

  std::unique_ptr<GPUExplicitState> gpu_state;

  bool
  try_gpu_explicit_step()
  {
    const int mode = ExperimentalGPU::explicit_mode();
    if (mode == 0)
      return false;
    const bool resident = (mode == 2);

    // Keep postprocessing (e.g. mg_n and f_tot) on its original CPU path.
    if (solve_block.solve_timing != SolveTiming::Primary)
      return false;

    const auto &pde = solve_context->get_pde_operator();
    const auto &attributes = solve_context->get_field_attributes();
    const auto &manager = solve_context->get_matrix_free_manager();
    if (!pde.supports_gpu_rhs(solve_block.id) ||
        (resident && !pde.supports_gpu_resident_explicit(solve_block.id)) ||
        solve_block.field_indices.size() != 1 ||
        solve_context->get_user_inputs().spatial_discretization.has_adaptivity ||
        solutions.num_levels() != 0 ||
        solutions.get_primary_solutions().old_solutions.size() != 1)
      throw std::runtime_error(
        "E1 GPU explicit requires a supported primary RHS, one scalar field, "
        "one old state, no AMR and no multigrid levels");

    const unsigned int field_index = *solve_block.field_indices.begin();
    if (attributes[field_index].field_type != TensorRank::Scalar)
      throw std::runtime_error("E1 GPU explicit supports scalar fields only");

    // The initial implementation deliberately excludes all affine constraints.
    // This covers the uniform-mesh Allen-Cahn example with natural BCs.
    // In particular, do not apply the LHS identity-row treatment to an RHS.
    if (manager.get_shared_matrix_free().get_affine_constraints(field_index)
          .n_constraints() != 0)
      throw std::runtime_error(
        "E1 GPU explicit currently requires zero affine constraints");

    const auto &mf = manager.get_portable_shared_matrix_free(field_index);
    const auto partitioner = mf.get_vector_partitioner();
    if (!partitioner || partitioner->n_mpi_processes() != 1 ||
        partitioner->n_ghost_indices() != 0 ||
        partitioner->n_import_indices() != 0)
      throw std::runtime_error(
        "E1 GPU explicit requires one MPI rank and zero ghost/import DoFs");

    // Age zero is the first old state (OldOne), not the current solution.
    const auto &host_old = solutions.get_old_solution_vector(0, field_index);
    auto &host_new = solutions.get_solution_vector(field_index);
    const auto &host_invm =
      solve_context->get_invm_manager().get_invm(TensorRank::Scalar);

    const std::size_t n = static_cast<std::size_t>(host_old.locally_owned_size());
    // With one MPI rank and full ownership, the local index equals the
    // global index. CPU and Portable MatrixFree must use the same DoF numbering.
    if (n == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        host_old.size() != n || host_new.size() != n || host_invm.size() != n ||
        host_new.locally_owned_size() != n || host_invm.locally_owned_size() != n)
      throw std::runtime_error("E1 GPU explicit host vector sizes do not match");

    // Fixed mesh: cache allocations and CPU-computed inverse mass once.
    // init()/reinit() invalidate this cache before any mesh reinitialization.
    const bool first_step = !gpu_state;
    if (first_step)
      {
        Timer::Scope setup_scope("E1 GPU Setup");
        auto state = std::make_unique<GPUExplicitState>();
        state->resident = resident;
        mf.initialize_dof_vector(state->old_solution);
        mf.initialize_dof_vector(state->new_solution);
        mf.initialize_dof_vector(state->inverse_mass);
        if (state->old_solution.locally_owned_size() != n ||
            state->old_solution.size() != n ||
            state->new_solution.locally_owned_size() != n ||
            state->inverse_mass.locally_owned_size() != n)
          throw std::runtime_error("E1 GPU explicit host/device sizes do not match");

        Kokkos::deep_copy(DeviceWrite(state->inverse_mass.get_values(), n),
                          HostRead(host_invm.get_values(), n));
        Kokkos::fence();
        gpu_state = std::move(state);

        std::cout << "[E1 GPU Explicit] Kokkos execution space: "
                  << ExecutionSpace::name() << std::endl;
        std::cout << "[E1 GPU Explicit] field: " << field_index
                  << ", DoFs: " << n
                  << (resident ? ", resident mode (device history + on-demand D2H)"
                               : ", bridge mode (H2D/D2H each step)") << std::endl;
      }

    auto &state = *gpu_state;
    if (state.resident != resident ||
        state.old_solution.get_partitioner().get() != partitioner.get() ||
        state.new_solution.get_partitioner().get() != partitioner.get() ||
        state.inverse_mass.get_partitioner().get() != partitioner.get() ||
        state.old_solution.locally_owned_size() != n)
      throw std::runtime_error("E1 GPU explicit cached layout changed");

    if (!resident || first_step)
    {
      Timer::Scope copy_scope("E1 H2D");
      Kokkos::deep_copy(DeviceWrite(state.old_solution.get_values(), n),
                        HostRead(host_old.get_values(), n));
      Kokkos::fence();
    }

    {
      // RHS includes n_old and dt, so the update is invm*RHS, not
      // n_old + dt*invm*RHS. compute_gpu_rhs() zeros dst exactly once.
      // Include the RHS zeroing/launch and inverse mass application in this
      // wall timer; the fences exclude transfers from the compute interval.
      Kokkos::fence();
      Timer::Scope compute_scope("E1 GPU RHS and InvM");
      if (!pde.compute_gpu_rhs(solve_block.id,
                               field_index,
                               manager,
                               solve_context->get_simulation_timer(),
                               state.new_solution,
                               state.old_solution))
        throw std::runtime_error("E1 GPU explicit PDE rejected GPU RHS");
      ExperimentalGPU::explicit_scale_inverse_mass(
        state.new_solution.get_values(), state.inverse_mass.get_values(),
        static_cast<int>(n));
      Kokkos::fence();
    }

    if (resident)
      {
        // Same idea as Pagoma's double-buffered device time iterator.
        // The compute interval above fences before these allocations swap.
        state.old_solution.swap(state.new_solution);
        state.has_result = true;
        state.host_current_valid = false;
      }
    else
      {
        Timer::Scope copy_scope("E1 D2H");
        Kokkos::deep_copy(HostWrite(host_new.get_values(), n),
                          DeviceRead(state.new_solution.get_values(), n));
        Kokkos::fence();
      }
    return true;
  }
#endif

  /**
   * @brief Matrix free operator.
   */
  MFOperator<dim, degree, number> rhs_operator;
};

PRISMS_PF_END_NAMESPACE
