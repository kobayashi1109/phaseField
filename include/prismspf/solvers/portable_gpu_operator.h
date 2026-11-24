#pragma once

// Experimental PRISMS-PF single-rank GPU matrix-free operator (deal.II 9.6).
// Generalizes the previously validated SingleGpuLaplaceOperator:
// the element operation is supplied as a template parameter.
// Intentionally bypasses Portable::MatrixFree::cell_loop() and MPI compress().
// NOT valid for multiple MPI ranks or nonzero ghost/import DoFs.

#ifndef PRISMS_PF_GPU
#  error "portable_gpu_operator.h requires PRISMS_PF_GPU"
#endif

#include <deal.II/base/memory_space.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

namespace prismspf
{
  // CellOperation must expose n_local_dofs and a device-callable operator():
  // operator()(unsigned int cell, const MatrixFree::Data*,
  //            Portable::SharedData<dim, Number>*, const Number*, Number*) const.
  template <int dim, int degree, typename Number, typename CellOperation>
  class SingleRankPortableOperator
  {
  public:
    using MatrixFree = dealii::Portable::MatrixFree<dim, Number>;
    using Vector = dealii::LinearAlgebra::distributed::Vector<
      Number, dealii::MemorySpace::Default>;
    using ExecutionSpace =
      typename dealii::MemorySpace::Default::kokkos_space::execution_space;
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TeamHandle = typename TeamPolicy::member_type;
    using ScratchView1D =
      Kokkos::View<Number *,
                   typename ExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchView2D =
      Kokkos::View<Number *[dim],
                   typename ExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    explicit SingleRankPortableOperator(const MatrixFree &matrix_free)
      : matrix_free_(matrix_free)
    {}

    // Public for CUDA/Kokkos-generated kernel registration code.
    struct DeviceCellKernel
    {
      typename MatrixFree::Data gpu_data;
      const Number             *src;
      Number                   *dst;
      CellOperation             cell_operation;

      DeviceCellKernel(const typename MatrixFree::Data &data,
                       const Number                   *src_ptr,
                       Number                         *dst_ptr,
                       const CellOperation &cell_operation_in = CellOperation{})
        : gpu_data(data)
        , src(src_ptr)
        , dst(dst_ptr)
        , cell_operation(cell_operation_in)
      {}

      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        return ScratchView1D::shmem_size(CellOperation::n_local_dofs) +
               ScratchView2D::shmem_size(CellOperation::n_local_dofs);
      }

      DEAL_II_HOST_DEVICE void
      operator()(const TeamHandle &team_member) const
      {
        ScratchView1D values(team_member.team_shmem(),
                             CellOperation::n_local_dofs);
        ScratchView2D gradients(team_member.team_shmem(),
                                CellOperation::n_local_dofs);

        dealii::Portable::SharedData<dim, Number> shared_data(
          team_member, values, gradients);

        cell_operation(team_member.league_rank(),
                       &gpu_data,
                       &shared_data,
                       src,
                       dst);
      }
    };

    // The caller zeros dst, deals with Dirichlet conditions, and fences as
    // necessary. No data transfer, MPI ghost exchange or compress is invoked.
    void
    apply_cells(Vector &dst,
                const Vector &src,
                const CellOperation &cell_operation = CellOperation{}) const
    {
      check_single_rank_layout(dst, src);

      const unsigned int n_colors = static_cast<unsigned int>(
        matrix_free_.get_colored_graph().size());

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          const auto gpu_data = matrix_free_.get_data(color);
          if (gpu_data.n_cells == 0)
            continue;

          const ExecutionSpace exec;
          const TeamPolicy policy(exec, gpu_data.n_cells, Kokkos::AUTO);
          const DeviceCellKernel kernel(gpu_data,
                                        src.get_values(),
                                        dst.get_values(),
                                        cell_operation);
          Kokkos::parallel_for("prismspf::single_rank_portable_operator",
                               policy,
                               kernel);
        }
    }

    // Standard operator interface: y = A*x, including constrained DoF copy.
    // The vectors remain resident on the GPU. No fence by design.
    void
    vmult(Vector &dst, const Vector &src) const
    {
      dst = Number(0);
      apply_cells(dst, src);
      matrix_free_.copy_constrained_values(src, dst);
    }

  private:
    void
    check_single_rank_layout(const Vector &dst, const Vector &src) const
    {
      const auto &partitioner = matrix_free_.get_vector_partitioner();
      if (!partitioner || partitioner->n_mpi_processes() != 1 ||
          partitioner->n_ghost_indices() != 0 ||
          partitioner->n_import_indices() != 0)
        throw std::runtime_error(
          "SingleRankPortableOperator requires one MPI rank and zero "
          "ghost/import DoFs; multi-rank use requires proper communication.");

      if (src.get_partitioner().get() != partitioner.get() ||
          dst.get_partitioner().get() != partitioner.get())
        throw std::runtime_error(
          "SingleRankPortableOperator requires matching src/dst partitioners.");
    }

    const MatrixFree &matrix_free_;
  };
} // namespace prismspf
