#pragma once

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/base/utilities.h>

namespace prismspf
{

template <int dim, int degree, typename Number>
struct PortableLaplaceQuadrature
{
  using FEEval =
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, Number>;

  DEAL_II_HOST_DEVICE void
  operator()(FEEval *fe_eval, const int q) const
  {
    fe_eval->submit_gradient(fe_eval->get_gradient(q), q);
  }
};

template <int dim, int degree, typename Number>
struct PortableLaplaceCell
{
  static constexpr unsigned int n_dofs_1d = degree + 1;
  static constexpr unsigned int n_local_dofs =
    dealii::Utilities::pow(degree + 1, dim);
  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  DEAL_II_HOST_DEVICE void
  operator()(
    const unsigned int,
    const typename dealii::Portable::MatrixFree<dim, Number>::Data *data,
    dealii::Portable::SharedData<dim, Number> *shared_data,
    const Number *src,
    Number *dst) const
  {
    using FEEval =
      dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, Number>;

    FEEval fe_eval(data, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(dealii::EvaluationFlags::gradients);

    fe_eval.apply_for_each_quad_point(
      PortableLaplaceQuadrature<dim, degree, Number>{});

    fe_eval.integrate(dealii::EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }
};

} // namespace prismspf
