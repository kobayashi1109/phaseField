#pragma once

// Experimental PRISMS-PF Stage D2-1: GPU-resident Jacobi-PCG.
// The inverse diagonal is evaluated by the existing CPU MFOperator and copied
// to the device once per solve. No full-vector copies during PCG iterations.
// Restricted to one MPI rank, one scalar block, zero ghost/import DoFs,
// homogeneous correction vectors, no AMR, SPD LHS, AbsoluteResidual.
// D1 remains in portable_native_cg.h and is not modified.

#ifndef PRISMS_PF_GPU
#  error "portable_native_pcg.h requires PRISMS_PF_GPU"
#endif

#include <deal.II/base/memory_space.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <prismspf/core/matrix_free_manager.h>
#include <prismspf/core/pde_operator_base.h>
#include <Kokkos_Core.hpp>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

PRISMS_PF_BEGIN_NAMESPACE

namespace ExperimentalGPU
{
  template <unsigned int dim, unsigned int degree, typename Number>
  bool
  native_pcg_solve(
    const PDEOperatorBase<dim, degree, Number> &pde,
    const MatrixFreeManager<dim, Number> &manager,
    unsigned int solve_block_id,
    unsigned int field_index,
    BlockVector<Number> &host_x,
    const BlockVector<Number> &host_b,
    const BlockVector<Number> &host_inverse_diagonal,
    dealii::SolverControl &control)
  {
    struct NVTXScope
    {
      explicit NVTXScope(const char *name) { nvtxRangePushA(name); }
      ~NVTXScope() { nvtxRangePop(); }
    } nvtx_scope("GPU_NATIVE_PCG_D2");

    using DeviceVector = typename PDEOperatorBase<dim, degree, Number>::DeviceVector;
    using DeviceSpace = dealii::MemorySpace::Default::kokkos_space;
    using ExecutionSpace = typename DeviceSpace::execution_space;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using DeviceWrite = Kokkos::View<Number *, DeviceSpace, Unmanaged>;
    using DeviceRead = Kokkos::View<const Number *, DeviceSpace, Unmanaged>;
    using HostWrite = Kokkos::View<Number *, Kokkos::HostSpace, Unmanaged>;
    using HostRead = Kokkos::View<const Number *, Kokkos::HostSpace, Unmanaged>;

    static bool printed_execution_space = false;
    if (!printed_execution_space)
      {
        std::cout << "[D2 GPU PCG] Kokkos execution space: "
                  << ExecutionSpace::name() << std::endl;
        printed_execution_space = true;
      }

    const auto &mf = manager.get_portable_shared_matrix_free(field_index);
    const auto partitioner = mf.get_vector_partitioner();
    if (!partitioner || partitioner->n_mpi_processes() != 1 ||
        partitioner->n_ghost_indices() != 0 ||
        partitioner->n_import_indices() != 0)
      throw std::runtime_error(
        "D2 GPU PCG requires one MPI rank with zero ghost/import DoFs");

    if (host_x.n_blocks() != 1 || host_b.n_blocks() != 1 ||
        host_inverse_diagonal.n_blocks() != 1)
      throw std::runtime_error("D2 GPU PCG requires one scalar field/block");

    DeviceVector x, b, r, z, p, Ap, Ax, inverse_diagonal;
    mf.initialize_dof_vector(x);
    mf.initialize_dof_vector(b);
    mf.initialize_dof_vector(r);
    mf.initialize_dof_vector(z);
    mf.initialize_dof_vector(p);
    mf.initialize_dof_vector(Ap);
    mf.initialize_dof_vector(Ax);
    mf.initialize_dof_vector(inverse_diagonal);

    const std::size_t n = static_cast<std::size_t>(x.locally_owned_size());
    // Under the one-rank restriction, all three host blocks must own the full
    // DoF range. As in D1, CPU and Portable MatrixFree use the same DoF numbering.
    if (host_x.block(0).locally_owned_size() != n ||
        host_b.block(0).locally_owned_size() != n ||
        host_inverse_diagonal.block(0).locally_owned_size() != n ||
        host_x.block(0).size() != n || host_b.block(0).size() != n ||
        host_inverse_diagonal.block(0).size() != n ||
        n == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("D2 GPU PCG host/device DoF sizes do not match");
    const Kokkos::RangePolicy<ExecutionSpace> range(0, static_cast<int>(n));

    // Copy x, b and the already inverted CPU diagonal once per solve.
    Kokkos::deep_copy(DeviceWrite(x.get_values(), n),
                      HostRead(host_x.block(0).get_values(), n));
    Kokkos::deep_copy(DeviceWrite(b.get_values(), n),
                      HostRead(host_b.block(0).get_values(), n));
    Kokkos::deep_copy(DeviceWrite(inverse_diagonal.get_values(), n),
                      HostRead(host_inverse_diagonal.block(0).get_values(), n));

    mf.set_constrained_values(Number(0), x);
    mf.set_constrained_values(Number(0), b);
    // Constrained residual entries are zero. Setting the corresponding
    // inverse diagonal entries to one is harmless and avoids 0*Inf/NaN.
    mf.set_constrained_values(Number(1), inverse_diagonal);
    Kokkos::fence();

    // Jacobi-PCG requires a positive, finite inverse diagonal on free DoFs.
    {
      const Number *d = inverse_diagonal.get_values();
      const Number largest = std::numeric_limits<Number>::max();
      int invalid = 0;
      Kokkos::parallel_reduce(
        "prismspf::native_pcg_validate_inverse_diagonal", range,
        KOKKOS_LAMBDA(const int i, int &count) {
          if (!(d[i] > Number(0)) || !(d[i] <= largest))
            ++count;
        }, invalid);
      if (invalid != 0)
        throw std::runtime_error("D2 GPU PCG inverse diagonal is not positive and finite");
    }

    auto dot = [&](const DeviceVector &u, const DeviceVector &v) -> double
    {
      const Number *u_ptr = u.get_values();
      const Number *v_ptr = v.get_values();
      double result = 0.0;
      Kokkos::parallel_reduce(
        "prismspf::native_pcg_dot", range,
        KOKKOS_LAMBDA(const int i, double &sum) {
          sum += static_cast<double>(u_ptr[i]) * static_cast<double>(v_ptr[i]);
        }, result);
      return result;
    };

    // Reuse the validated D1 GPU matrix-free path unchanged.
    auto matvec = [&](DeviceVector &dst, const DeviceVector &src)
    {
      dst = Number(0);
      if (!pde.compute_gpu_lhs(solve_block_id, field_index, manager, dst, src))
        throw std::runtime_error("D2 GPU PCG PDE rejected GPU LHS operation");
      mf.set_constrained_values(Number(0), dst);
    };

    // New D2 operation: z = D^{-1} r. All arrays are on the device.
    auto apply_jacobi = [&]()
    {
      Number *z_ptr = z.get_values();
      const Number *r_ptr = r.get_values();
      const Number *d_ptr = inverse_diagonal.get_values();
      Kokkos::parallel_for(
        "prismspf::native_pcg_apply_jacobi", range,
        KOKKOS_LAMBDA(const int i) { z_ptr[i] = d_ptr[i] * r_ptr[i]; });
    };

    // r0 = b - Ax0. Initial convergence check uses the UNpreconditioned residual.
    matvec(Ax, x);
    {
      Number *r_ptr = r.get_values();
      const Number *b_ptr = b.get_values();
      const Number *Ax_ptr = Ax.get_values();
      Kokkos::parallel_for(
        "prismspf::native_pcg_initial_residual", range,
        KOKKOS_LAMBDA(const int i) { r_ptr[i] = b_ptr[i] - Ax_ptr[i]; });
    }
    mf.set_constrained_values(Number(0), r);

    double rr = dot(r, r);
    if (!std::isfinite(rr) || rr < 0.0)
      throw std::runtime_error("D2 GPU PCG initial residual is invalid");
    auto state = control.check(0, std::sqrt(rr));
    unsigned int iteration = 0;
    double rho = 0.0;

    if (state == dealii::SolverControl::iterate)
      {
        apply_jacobi();
        // p0 = z0
        Kokkos::deep_copy(DeviceWrite(p.get_values(), n),
                          DeviceRead(z.get_values(), n));
        rho = dot(r, z);
        if (!(rho > 0.0) || !std::isfinite(rho))
          throw std::runtime_error("D2 GPU PCG initial r^T z is not positive and finite");
      }

    // Separate NVTX range for the PCG iteration loop, without allocations,
    // host-device vector copies or initial residual evaluation.
    Kokkos::fence();
    {
      NVTXScope iteration_scope("GPU_PCG_ITERATIONS_D2");
      while (state == dealii::SolverControl::iterate)
        {
          ++iteration;
          matvec(Ap, p);
          const double pAp = dot(p, Ap);
          if (!(pAp > 0.0) || !std::isfinite(pAp))
            throw std::runtime_error("D2 GPU PCG breakdown: invalid p^T A p");

          const Number alpha = static_cast<Number>(rho / pAp);
          if (!std::isfinite(alpha))
            throw std::runtime_error("D2 GPU PCG breakdown: invalid alpha");
          {
            Number *x_ptr = x.get_values();
            Number *r_ptr = r.get_values();
            const Number *p_ptr = p.get_values();
            const Number *Ap_ptr = Ap.get_values();
            Kokkos::parallel_for(
              "prismspf::native_pcg_update_x_r", range,
              KOKKOS_LAMBDA(const int i) {
                x_ptr[i] += alpha * p_ptr[i];
                r_ptr[i] -= alpha * Ap_ptr[i];
              });
          }

          rr = dot(r, r);
          if (!std::isfinite(rr) || rr < 0.0)
            throw std::runtime_error("D2 GPU PCG residual became invalid");
          state = control.check(iteration, std::sqrt(rr));
          if (state != dealii::SolverControl::iterate)
            break;

          apply_jacobi();
          const double rho_new = dot(r, z);
          if (!(rho_new > 0.0) || !std::isfinite(rho_new))
            throw std::runtime_error("D2 GPU PCG breakdown: invalid r^T z");

          const Number beta = static_cast<Number>(rho_new / rho);
          if (!std::isfinite(beta))
            throw std::runtime_error("D2 GPU PCG breakdown: invalid beta");
          {
            Number *p_ptr = p.get_values();
            const Number *z_ptr = z.get_values();
            Kokkos::parallel_for(
              "prismspf::native_pcg_update_p", range,
              KOKKOS_LAMBDA(const int i) {
                p_ptr[i] = z_ptr[i] + beta * p_ptr[i];
              });
          }
          rho = rho_new;
        }
      Kokkos::fence();
    }

    // Existing PRISMS-PF solution management stays on the host in D2-1.
    Kokkos::deep_copy(HostWrite(host_x.block(0).get_values(), n),
                      DeviceRead(x.get_values(), n));
    return state == dealii::SolverControl::success;
  }
} // namespace ExperimentalGPU

PRISMS_PF_END_NAMESPACE
