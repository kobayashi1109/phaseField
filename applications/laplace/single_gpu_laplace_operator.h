#pragma once

// PRISMS-PF GPU probe (deal.II 9.6.0 + Kokkos 4.2)
// A single-MPI-rank GPU-only dispatch of the existing PortableLaplaceCell.
// Unlike Portable::MatrixFree::cell_loop() on distributed meshes, this path
// deliberately does NOT call update_ghost_values() or compress().
// It is NOT a replacement for MPI-enabled multi-GPU execution.

#include "portable_laplace_probe.h"

#include <deal.II/base/memory_space.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

namespace prismspf
{
  template <int dim, int degree, typename Number>
  class SingleGpuLaplaceOperator
  {
  public:
    using MatrixFree = dealii::Portable::MatrixFree<dim, Number>;
    using Vector = dealii::LinearAlgebra::distributed::Vector<
      Number, dealii::MemorySpace::Default>;
    using CellOperation = PortableLaplaceCell<dim, degree, Number>;
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

    explicit SingleGpuLaplaceOperator(const MatrixFree &matrix_free)
      : matrix_free_(matrix_free)
    {}

    // Apply ONLY the element operator to an already-zeroed device vector.
    // This does not update ghost values or perform MPI compression.
    // Kokkos kernel launches are asynchronous: caller must fence before timing
    // or reading the result.
    void
    apply_cells(Vector &dst, const Vector &src) const
    {
      check_single_gpu_layout(dst, src);

      const unsigned int n_colors =
        static_cast<unsigned int>(matrix_free_.get_colored_graph().size());

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          const auto gpu_data = matrix_free_.get_data(color);
          if (gpu_data.n_cells == 0)
            continue;

          const ExecutionSpace exec;
          const TeamPolicy policy(exec, gpu_data.n_cells, Kokkos::AUTO);
          const DeviceCellKernel kernel(gpu_data,
                                        src.get_values(),
                                        dst.get_values());

          // Direct device launch: no call to MatrixFree::cell_loop(),
          // distributed::Vector::compress(), or ghost exchange.
          Kokkos::parallel_for("prismspf::single_gpu_laplace_cells",
                               policy,
                               kernel);
        }
    }

    // Pagoma-style interface. The entire operator application remains on GPU
    // once the vectors have been initialized and populated.
    void
    vmult(Vector &dst, const Vector &src) const
    {
      dst = Number(0);
      apply_cells(dst, src);
      matrix_free_.copy_constrained_values(src, dst);
      // No fence here: let the calling algorithm control synchronization.
    }

  public:
    struct DeviceCellKernel
    {
      typename MatrixFree::Data gpu_data;
      const Number             *src;
      Number                   *dst;

      DeviceCellKernel(const typename MatrixFree::Data &gpu_data_in,
                       const Number                   *src_in,
                       Number                         *dst_in)
        : gpu_data(gpu_data_in)
        , src(src_in)
        , dst(dst_in)
      {}

      // Kokkos invokes this on the host to determine scratch requirements.
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

        CellOperation{}(team_member.league_rank(),
                        &gpu_data,
                        &shared_data,
                        src,
                        dst);
      }
    };

  private:
    void
    check_single_gpu_layout(const Vector &dst, const Vector &src) const
    {
      const auto &partitioner = matrix_free_.get_vector_partitioner();

      if (!partitioner || partitioner->n_mpi_processes() != 1 ||
          partitioner->n_ghost_indices() != 0 ||
          partitioner->n_import_indices() != 0)
        throw std::runtime_error(
          "SingleGpuLaplaceOperator requires exactly one MPI rank and "
          "zero ghost/import DoFs. Multi-rank execution needs a separate "
          "correct MPI communication path.");

      if (src.get_partitioner().get() != partitioner.get() ||
          dst.get_partitioner().get() != partitioner.get())
        throw std::runtime_error(
          "SingleGpuLaplaceOperator: input/output partitioners must match "
          "the Portable::MatrixFree partitioner.");
    }

    const MatrixFree &matrix_free_;
  };
} // namespace prismspf
