#pragma once

// Experimental single-field, single-MPI-rank GPU PDE operator for PRISMS-PF.
// Connects MatrixFreeManager (phase B) with the device-callable scalar PDE
// interface and the direct Kokkos matrix-free kernel (phase C).
//
// Requires: PRISMS_PF_GPU, deal.II Portable::MatrixFree, and a device-callable
// PDE functor with evaluation_flags, integration_flags, operator()(field).
//
// Limitations: one scalar field; one MPI rank; no ghost/import DoFs;
// no CPU BlockVector adapter, multigrid, diagonal evaluation or native GPU CG.
// The existing SingleRankPortableOperator validates MPI/vector partitioners.

#ifndef PRISMS_PF_GPU
#  error "portable_scalar_pde_operator.h requires PRISMS_PF_GPU"
#endif

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <prismspf/core/matrix_free_manager.h>
#include <prismspf/solvers/portable_gpu_operator.h>
#include <prismspf/solvers/portable_scalar_pde.h>

namespace prismspf
{
  template <int dim, int degree, typename Number, typename DevicePDE>
  class PortableScalarPDEOperator
  {
  public:
    using MatrixFree = dealii::Portable::MatrixFree<dim, Number>;
    using Vector = typename SingleRankPortableOperator<
      dim,
      degree,
      Number,
      PortableScalarPDECell<dim, degree, Number, DevicePDE>>::Vector;
    using Cell = PortableScalarPDECell<dim, degree, Number, DevicePDE>;
    using KernelOperator = SingleRankPortableOperator<dim, degree, Number, Cell>;

    PortableScalarPDEOperator(const MatrixFreeManager<dim, Number> &manager,
                             const unsigned int                    field_index)
      : matrix_free_(manager.get_portable_shared_matrix_free(field_index))
      , kernel_operator_(matrix_free_)
    {}

    // Allocates a GPU-resident vector with this field's partitioner.
    void
    initialize_dof_vector(Vector &vector) const
    {
      matrix_free_.initialize_dof_vector(vector);
    }

    // Full matrix-vector product. dst is zeroed and constrained values copied
    // entirely on the GPU. No explicit synchronization; caller fences if needed.
    void
    vmult(Vector &dst, const Vector &src) const
    {
      kernel_operator_.vmult(dst, src);
    }

    // Low-level cell integral only. Caller zeroes dst and applies constraints.
    // Retained to preserve the existing Laplace benchmark and NVTX ranges.
    void
    apply_cells(Vector &dst,
                const Vector &src,
                const DevicePDE &pde = DevicePDE{}) const
    {
      kernel_operator_.apply_cells(dst, src, Cell{pde});
    }

    [[nodiscard]] const MatrixFree &
    get_matrix_free() const
    {
      return matrix_free_;
    }

  private:
    const MatrixFree &matrix_free_;
    KernelOperator   kernel_operator_;
  };
} // namespace prismspf
