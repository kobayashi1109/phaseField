#pragma once

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <utility>

namespace prismspf
{

template <int dim, int degree, typename Number>
struct CpuLaplaceCell
{
  using HostVector =
    dealii::LinearAlgebra::distributed::Vector<Number>;

  void apply(
    const dealii::MatrixFree<dim, Number> &data,
    HostVector &dst,
    const HostVector &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    dealii::FEEvaluation<dim, degree, degree + 1, 1, Number>
      fe_eval(data, 0);

    for (unsigned int cell = cell_range.first;
         cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(dealii::EvaluationFlags::gradients);

      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
      }

      fe_eval.integrate(dealii::EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }
  }
};

} // namespace prismspf
