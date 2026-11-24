#pragma once

// Experimental single-GPU, single-scalar-field conjugate-gradient solver.
// PRISMS-PF gpu-dev + Stage C host-dispatched GPU LHS callback, deal.II 9.6.
//
// The CG vectors (x, b, r, p, Ap, Ax) stay resident on the GPU throughout
// the iteration. Only scalar dot products / convergence checks synchronize
// with the CPU each iteration. Full vectors are copied at entry/exit only.
//
// Requirements: 1 MPI rank; zero ghosts/imports; no adaptive hanging nodes;
// symmetric positive definite LHS; no preconditioner; AbsoluteResidual.
// The caller enforces the PDE, solve-block and parameter restrictions.

#ifndef PRISMS_PF_GPU
#  error "portable_native_cg.h requires PRISMS_PF_GPU"
#endif

#include <deal.II/base/memory_space.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
// Required for set_constrained_values templated implementation in deal.II 9.6.
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <prismspf/core/matrix_free_manager.h>
#include <prismspf/core/pde_operator_base.h>

#include <Kokkos_Core.hpp>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

PRISMS_PF_BEGIN_NAMESPACE

namespace ExperimentalGPU
{
  template <unsigned int dim, unsigned int degree, typename Number>
  bool
  native_cg_solve(
    const PDEOperatorBase<dim, degree, Number> &pde,
    const MatrixFreeManager<dim, Number> &manager,
    unsigned int solve_block_id,
    unsigned int field_index,
    BlockVector<Number> &host_x,
    const BlockVector<Number> &host_b,
    dealii::SolverControl &control)
  {
    struct NVTXScope
    {
      NVTXScope() { nvtxRangePushA("GPU_NATIVE_CG_D1"); }
      ~NVTXScope() { nvtxRangePop(); }
    } nvtx_scope;

    using DeviceVector = typename PDEOperatorBase<dim, degree, Number>::DeviceVector;
    using DeviceSpace = dealii::MemorySpace::Default::kokkos_space;
    using ExecutionSpace = typename DeviceSpace::execution_space;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using DeviceWrite = Kokkos::View<Number *, DeviceSpace, Unmanaged>;
    using DeviceRead = Kokkos::View<const Number *, DeviceSpace, Unmanaged>;
    using HostWrite = Kokkos::View<Number *, Kokkos::HostSpace, Unmanaged>;
    using HostRead = Kokkos::View<const Number *, Kokkos::HostSpace, Unmanaged>;

    const auto &mf = manager.get_portable_shared_matrix_free(field_index);
    const auto partitioner = mf.get_vector_partitioner();
    if (!partitioner || partitioner->n_mpi_processes() != 1 ||
        partitioner->n_ghost_indices() != 0 ||
        partitioner->n_import_indices() != 0)
      throw std::runtime_error(
        "GPU-native CG requires one MPI rank with zero ghost/import DoFs");

    if (host_x.n_blocks() != 1 || host_b.n_blocks() != 1)
      throw std::runtime_error("GPU-native CG requires one scalar field/block");

    DeviceVector x, b, r, p, Ap, Ax;
    mf.initialize_dof_vector(x);
    mf.initialize_dof_vector(b);
    mf.initialize_dof_vector(r);
    mf.initialize_dof_vector(p);
    mf.initialize_dof_vector(Ap);
    mf.initialize_dof_vector(Ax);

    const std::size_t n = static_cast<std::size_t>(x.locally_owned_size());
    if (host_x.block(0).locally_owned_size() != n ||
        host_b.block(0).locally_owned_size() != n ||
        n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("GPU-native CG local DoF sizes do not match");
    const int n_int = static_cast<int>(n);
    const Kokkos::RangePolicy<ExecutionSpace> range(0, n_int);

    // Two initial host -> GPU vector transfers; no full-vector transfers in CG.
    Kokkos::deep_copy(DeviceWrite(x.get_values(), n),
                      HostRead(host_x.block(0).get_values(), n));
    Kokkos::deep_copy(DeviceWrite(b.get_values(), n),
                      HostRead(host_b.block(0).get_values(), n));

    // Homogeneous Dirichlet treatment, matching the Stage C MFOperator bridge.
    // This first version intentionally excludes adaptive hanging-node problems.
    mf.set_constrained_values(Number(0), x);
    mf.set_constrained_values(Number(0), b);

    auto dot = [&](const DeviceVector &u, const DeviceVector &v) -> double
    {
      const Number *u_ptr = u.get_values();
      const Number *v_ptr = v.get_values();
      double result = 0.0;
      Kokkos::parallel_reduce(
        "prismspf::native_cg_dot",
        range,
        KOKKOS_LAMBDA(const int i, double &local_sum) {
          local_sum += static_cast<double>(u_ptr[i]) *
                       static_cast<double>(v_ptr[i]);
        },
        result);
      // Host scalar result requires synchronization, but no vector copy.
      return result;
    };

    auto matvec = [&](DeviceVector &dst, const DeviceVector &src)
    {
      dst = Number(0);
      const bool supported =
        pde.compute_gpu_lhs(solve_block_id, field_index, manager, dst, src);
      if (!supported)
        throw std::runtime_error("GPU-native CG PDE rejected GPU LHS operation");
      mf.set_constrained_values(Number(0), dst);
    };

    // r0 = b - A*x0, p0 = r0.
    matvec(Ax, x);
    {
      Number *r_ptr = r.get_values();
      Number *p_ptr = p.get_values();
      const Number *b_ptr = b.get_values();
      const Number *Ax_ptr = Ax.get_values();
      Kokkos::parallel_for(
        "prismspf::native_cg_initial_residual",
        range,
        KOKKOS_LAMBDA(const int i) {
          const Number ri = b_ptr[i] - Ax_ptr[i];
          r_ptr[i] = ri;
          p_ptr[i] = ri;
        });
    }
    mf.set_constrained_values(Number(0), r);
    mf.set_constrained_values(Number(0), p);

    double rr = dot(r, r);
    if (!std::isfinite(rr) || rr < 0.0)
      throw std::runtime_error("GPU-native CG initial residual is invalid");

    auto state = control.check(0, std::sqrt(rr));
    unsigned int iteration = 0;

    while (state == dealii::SolverControl::iterate)
      {
        ++iteration;
        matvec(Ap, p);
        const double pAp = dot(p, Ap);
        if (!(pAp > 0.0) || !std::isfinite(pAp))
          throw std::runtime_error(
            "GPU-native CG breakdown (p^T A p <= 0 or not finite)");

        const Number alpha = static_cast<Number>(rr / pAp);
        {
          Number *x_ptr = x.get_values();
          Number *r_ptr = r.get_values();
          const Number *p_ptr = p.get_values();
          const Number *Ap_ptr = Ap.get_values();
          Kokkos::parallel_for(
            "prismspf::native_cg_update_x_r",
            range,
            KOKKOS_LAMBDA(const int i) {
              x_ptr[i] += alpha * p_ptr[i];
              r_ptr[i] -= alpha * Ap_ptr[i];
            });
        }

        const double rr_new = dot(r, r);
        if (!std::isfinite(rr_new) || rr_new < 0.0)
          throw std::runtime_error("GPU-native CG residual became invalid");
        state = control.check(iteration, std::sqrt(rr_new));
        if (state != dealii::SolverControl::iterate)
          break;

        const Number beta = static_cast<Number>(rr_new / rr);
        {
          Number *p_ptr = p.get_values();
          const Number *r_ptr = r.get_values();
          Kokkos::parallel_for(
            "prismspf::native_cg_update_p",
            range,
            KOKKOS_LAMBDA(const int i) {
              p_ptr[i] = r_ptr[i] + beta * p_ptr[i];
            });
        }
        rr = rr_new;
      }

    // One final GPU -> host vector transfer for PRISMS-PF's existing
    // CPU solution manager, output, and time-stepping logic.
    Kokkos::fence();
    Kokkos::deep_copy(HostWrite(host_x.block(0).get_values(), n),
                      DeviceRead(x.get_values(), n));
    return state == dealii::SolverControl::success;
  }
} // namespace ExperimentalGPU

PRISMS_PF_END_NAMESPACE
