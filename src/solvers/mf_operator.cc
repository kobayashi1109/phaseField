// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <prismspf/core/exceptions.h>
#include <prismspf/core/field_container.h>
#include <prismspf/core/group_solution_handler.h>

#include <prismspf/solvers/mf_operator.h>

#include <prismspf/utilities/assert.h>


#ifdef PRISMS_PF_GPU
#  include <deal.II/matrix_free/portable_matrix_free.h>
#  include <deal.II/matrix_free/portable_matrix_free.templates.h>
#  include <deal.II/base/mpi.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <Kokkos_Core.hpp>
#  include <algorithm>
#  include <cmath>
#  include <cstdlib>
#  include <cstring>
#  include <iostream>
#endif

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_operator(BlockVector<number>       &dst,
                                                  const BlockVector<number> &src) const
{
  Timer::Scope scope("Compute Operator");

  bool used_gpu = false;
#ifdef PRISMS_PF_GPU
  const char *gpu_env = std::getenv("PRISMS_PF_C_GPU_VMULT");
  const bool gpu_requested = gpu_env != nullptr && std::strcmp(gpu_env, "1") == 0;
  using PDEBase = PDEOperatorBase<dim, degree, number>;
  if (gpu_requested && !read_plain && pde_op == &PDEBase::compute_lhs &&
      solve_block.field_indices.size() == 1 && src.n_blocks() == 1 &&
      dst.n_blocks() == 1 && pde_operator != nullptr &&
      pde_operator->supports_gpu_lhs(solve_block.id))
    {
      const unsigned int field_index = *solve_block.field_indices.begin();
      const auto &portable_mf =
        matrix_free_manager->get_portable_shared_matrix_free(field_index);
      const auto partitioner = portable_mf.get_vector_partitioner();
      AssertThrow(partitioner != nullptr && partitioner->n_mpi_processes() == 1 &&
                    partitioner->n_ghost_indices() == 0 &&
                    partitioner->n_import_indices() == 0,
                  dealii::ExcMessage("C GPU vmult supports only one MPI rank with zero ghosts/imports"));

      using DeviceVector = typename PDEBase::DeviceVector;
      DeviceVector device_src, device_dst;
      portable_mf.initialize_dof_vector(device_src);
      portable_mf.initialize_dof_vector(device_dst);
      const unsigned int n = device_src.locally_owned_size();
      AssertThrow(src.block(0).locally_owned_size() == n &&
                    dst.block(0).locally_owned_size() == n,
                  dealii::ExcMessage("Host/device owned DoF counts differ"));

      using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
      using DeviceMemory = dealii::MemorySpace::Default::kokkos_space;
      using HostRead = Kokkos::View<const number *, Kokkos::HostSpace, Unmanaged>;
      using HostWrite = Kokkos::View<number *, Kokkos::HostSpace, Unmanaged>;
      using DeviceRead = Kokkos::View<const number *, DeviceMemory, Unmanaged>;
      using DeviceWrite = Kokkos::View<number *, DeviceMemory, Unmanaged>;

      Kokkos::deep_copy(DeviceWrite(device_src.get_values(), n),
                        HostRead(src.block(0).get_values(), n));

      // C4_GPU_DIRICHLET_INPUT_ZERO_20261117
      // CPU FEEvaluation::read_dof_values() applies homogeneous constraints.
      // Portable::FEEvaluation gathers raw constrained entries. The standalone
      // Laplace probe explicitly zeroed them; the CG solver generally does not.
      // This C-stage path handles single-rank homogeneous constraints only.
      portable_mf.set_constrained_values(number(0), device_src);
      Kokkos::fence(); // Finish masking before entering the custom GPU operator.

      // Diagnostic: whether the CPU solver supplied nonzero constrained DoFs.
      static bool printed_constrained_input = false;
      if (!printed_constrained_input)
        {
          number constrained_source_sq = number(0);
          for (const auto i : data->get_constrained_dofs(field_index))
            if (i < n)
              {
                const number v = src.block(0).local_element(i);
                constrained_source_sq += v * v;
              }
          std::cout << "[GPU MFOperator] constrained src L2 norm before masking: "
                    << std::sqrt(constrained_source_sq) << std::endl;
          printed_constrained_input = true;
        }

      used_gpu = pde_operator->compute_gpu_lhs(
        solve_block.id, field_index, *matrix_free_manager, device_dst, device_src);
      if (used_gpu)
        {
          Kokkos::fence(); // Ensure cell accumulation is complete.
          // Match CPU integrate_scatter: constrained rows must be zero.
          portable_mf.set_constrained_values(number(0), device_dst);
          Kokkos::fence();
          Kokkos::deep_copy(HostWrite(dst.block(0).get_values(), n),
                            DeviceRead(device_dst.get_values(), n));

          static bool printed = false;
          if (!printed)
            {
              std::cout << "[GPU MFOperator] C-stage host/device adapter active "
                           "(CPU CG, GPU LHS matvec)" << std::endl;
              printed = true;
            }

          const char *verify = std::getenv("PRISMS_PF_C_GPU_VERIFY");
          static bool checked = false;
          if (!checked && verify != nullptr && std::strcmp(verify, "1") == 0)
            {
              BlockVector<number> cpu_reference;
              cpu_reference.reinit(dst);
              data->cell_loop(&MFOperator::compute_local_operator,
                              this, cpu_reference, src, true);
              BlockVector<number> difference;
              difference.reinit(dst);
              difference = dst;
              difference -= cpu_reference;
              const number relative_error = difference.l2_norm() /
                std::max(cpu_reference.l2_norm(), number(1.0e-30));
              std::cout << "[GPU MFOperator] CPU/GPU vmult relative L2 error: "
                        << relative_error << std::endl;
              AssertThrow(relative_error < number(1.0e-12),
                          dealii::ExcMessage("GPU MFOperator vmult differs from CPU"));
              checked = true;
            }
        }
    }
#endif  // PRISMS_PF_C_GPU_VMULT
  if (!used_gpu)
    data->cell_loop(&MFOperator::compute_local_operator, this, dst, src, true);
  if (scale_by_diagonal)
    {
      for (unsigned int block_index = 0; block_index < dst.n_blocks(); block_index++)
        {
          dst.block(block_index).scale(*(scaling_diagonal[block_index]));
        }
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_local_operator(
  const MatrixFree<dim, number>               &_data,
  BlockVector<number>                         &dst,
  const BlockVector<number>                   &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Construct FEEvaluation objects
  // The reason this is constructed here, rather than as a private member is because
  // compute_local_rhs is called by cell_loop, which multithreads. There would be data
  // races.
  FieldContainer<dim, degree, number> variable_list(field_attributes,
                                                    *solution_indexer,
                                                    relative_level,
                                                    dependency_map,
                                                    solve_block,
                                                    _data);

  // Initialize, evaluate, and submit based on user function.
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // Initialize, read DOFs, and set evaluation flags for each variable
      variable_list.reinit_and_eval(cell, &src, read_plain);

      // Evaluate the user-defined pde at each quadrature point
      for (unsigned int quad = 0; quad < variable_list.get_n_q_points(); ++quad)
        {
          variable_list.set_q_point(quad);
          // Evaluate the function pointer (the user-defined pde)
          try
            {
              (pde_operator->*pde_op)(variable_list, *sim_timer, solve_block.id);
            }
          catch (...)
            {
              std::cerr << "Error: Exception thrown in equations during solve block "
                        << solve_block.id << "!" << std::endl;
              throw;
            }
        }

      // Integrate and add to global vector dst

      variable_list.integrate_and_distribute(&dst);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_diagonal(BlockVector<number>       &dst,
                                                  const BlockVector<number> &src) const
{
  dst.reinit(src);
  data->cell_loop(&MFOperator::compute_local_diagonal, this, dst, src);
  if (scale_by_diagonal)
    {
      for (unsigned int block_index = 0; block_index < dst.n_blocks(); block_index++)
        {
          dst.block(block_index).scale(*(scaling_diagonal[block_index]));
        }
    }
  // This is to make sure the preconditioner doesn't break down when there are
  // Dirichlet conditions (which lead to zero diagonal entries). The actual value
  // of these entries doesn't matter since they get overwritten by the constraints
  // application in the solver.
  set_zero_entries_to_one(dst);
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::compute_local_diagonal(
  const MatrixFree<dim, number>               &_data,
  BlockVector<number>                         &diagonal,
  const BlockVector<number>                   &dummy_src, // just needs right shape
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FieldContainer<dim, degree, number> variable_list(field_attributes,
                                                    *solution_indexer,
                                                    relative_level,
                                                    dependency_map,
                                                    solve_block,
                                                    _data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // Reinit the cell for all the dependencies
      variable_list.reinit_and_eval(cell, &dummy_src, false);

      // To get the diagonal of the "matrix", repeatedly "multiply the matrix" by a test x
      // vector (src vector) and store the solution in the i-th position in the y
      // vector (the diagonal).
      // First (i=1) x vector: I 0 0 0
      // Second(i=2) x vector: 0 I 0 0 ...
      // Submit zeros for everything except the diagonals
      for (unsigned int field_index : solve_block.field_indices)
        {
          if (field_attributes[field_index].field_type == TensorRank::Scalar)
            {
              compute_local_field_diagonal<TensorRank::Scalar>(variable_list,
                                                               diagonal,
                                                               field_index);
            }
          else if (field_attributes[field_index].field_type == TensorRank::Vector)
            {
              compute_local_field_diagonal<TensorRank::Vector>(variable_list,
                                                               diagonal,
                                                               field_index);
            }
        }
    }
}

template <unsigned int dim, unsigned int degree, typename number>
template <TensorRank Rank>
void
MFOperator<dim, degree, number>::compute_local_field_diagonal(
  FieldContainer<dim, degree, number> &variable_list,
  BlockVector<number>                 &diagonal,
  unsigned int                         field_index) const
{
  // Number of nodes in the cell
  constexpr static unsigned int dofs_per_component =
    FieldContainer<dim, degree, number>::dofs_per_component;
  // "zero lhs vector"
  for (unsigned int some_field_index : solve_block.field_indices)
    {
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        {
          if (field_attributes[some_field_index].field_type == TensorRank::Scalar)
            {
              variable_list.submit_dof_value(some_field_index,
                                             zero<TensorRank::Scalar>(),
                                             i);
            }
          else if (field_attributes[some_field_index].field_type == TensorRank::Vector)
            {
              variable_list.submit_dof_value(some_field_index,
                                             zero<TensorRank::Vector>(),
                                             i);
            }
        }
    }
  // Object to hold the local diagonal
  dealii::AlignedVector<Value<Rank>> cell_diagonal(dofs_per_component, zero<Rank>());
  for (unsigned int i = 0; i < dofs_per_component; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_component; ++j)
        {
          variable_list.submit_dof_value(field_index,
                                         i == j ? identity<Rank>() : zero<Rank>(),
                                         j);
        }

      // Evaluate the dependencies based on the flags
      variable_list.eval_without_read();

      // Evaluate at each quadrature point
      for (unsigned int quad = 0; quad < variable_list.get_n_q_points(); ++quad)
        {
          variable_list.set_q_point(quad);
          try
            {
              (pde_operator->*pde_op)(variable_list, *sim_timer, solve_block.id);
            }
          catch (...)
            {
              std::cerr << "Error: Exception thrown in equations during solve block "
                        << solve_block.id << "!" << std::endl;
              throw;
            }
        }
      // Integrate the diagonal
      variable_list.integrate();

      // variable_list.eval(); //needed?
      variable_list.get_dof_value_to(cell_diagonal[i], field_index, i);
    }
  // Use FieldContainer interface to write local diagonal to global
  for (unsigned int i = 0; i < dofs_per_component; ++i)
    {
      variable_list.submit_dof_value(field_index, cell_diagonal[i], i);
    }
  variable_list.distribute(field_index, &diagonal);
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::reinit_matrix_diagonal()
{
  if (relative_level == -1)
    {
      matrix_free_manager->initialize_block_vector(diagonal_entries->get_vector(),
                                                   solve_block.field_indices);
      matrix_free_manager->initialize_block_vector(inverse_diagonal_entries->get_vector(),
                                                   solve_block.field_indices);
    }
  else
    {
      matrix_free_manager->initialize_mg_block_vector(diagonal_entries->get_vector(),
                                                      solve_block.field_indices,
                                                      relative_level);
      matrix_free_manager->initialize_mg_block_vector(
        inverse_diagonal_entries->get_vector(),
        solve_block.field_indices,
        relative_level);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::eval_matrix_diagonal()
{
  auto &diag     = diagonal_entries->get_vector();
  auto &inv_diag = inverse_diagonal_entries->get_vector();
  compute_diagonal(diag, inv_diag); // using inv_diag as dummy
  for (unsigned int i : diag.locally_owned_elements())
    {
      number diag_el = diag[i];
      inv_diag[i]    = 1.0 / diag_el;
    }
}

template <unsigned int dim, unsigned int degree, typename number>
dealii::types::global_dof_index
MFOperator<dim, degree, number>::m() const
{
  Assert(data != nullptr, dealii::ExcNotInitialized());
  unsigned int sum = 0;
  for (unsigned int field_index : solve_block.field_indices)
    {
      sum += data->get_vector_partitioner(field_index)->size();
    }
  return sum;
}

template <unsigned int dim, unsigned int degree, typename number>
number
MFOperator<dim, degree, number>::el([[maybe_unused]] const unsigned int &row,
                                    [[maybe_unused]] const unsigned int &col) const
{
  ASSERT(false, "el() hasn't been implemented");
  return 0.0;
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::clear()
{
  data = nullptr;
  diagonal_entries.reset();
  inverse_diagonal_entries.reset();
}

template <unsigned int dim, unsigned int degree, typename number>
const MatrixFree<dim, number> *
MFOperator<dim, degree, number>::get_matrix_free() const
{
  return data;
}

template <unsigned int dim, unsigned int degree, typename number>
const std::shared_ptr<dealii::DiagonalMatrix<BlockVector<number>>> &
MFOperator<dim, degree, number>::get_matrix_diagonal_inverse() const
{
  Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
         dealii::ExcNotInitialized());
  return inverse_diagonal_entries;
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::vmult(BlockVector<number>       &dst,
                                       const BlockVector<number> &src) const
{
  compute_operator(dst, src);
}

template <unsigned int dim, unsigned int degree, typename number>
void
MFOperator<dim, degree, number>::Tvmult(BlockVector<number>       &dst,
                                        const BlockVector<number> &src) const
{
  vmult(dst, src);
}

#include "solvers/mf_operator.inst"

PRISMS_PF_END_NAMESPACE
