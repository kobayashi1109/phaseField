#pragma once

// Experimental PRISMS-PF GPU scalar PDE interface (deal.II 9.6).
// Supports one scalar field, with value and/or gradient terms.
// Designed for SingleRankPortableOperator; not a replacement for the
// CPU FieldContainer or for multi-field, multi-rank PDE assembly.

#ifndef PRISMS_PF_GPU
#  error "portable_scalar_pde.h requires PRISMS_PF_GPU"
#endif

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace prismspf
{
  // Lightweight view of one quadrature point. No allocation or host transfer.
  template <int dim, int degree, typename Number>
  class PortableScalarField
  {
  public:
    using FEEval =
      dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, Number>;
    using Gradient = dealii::Tensor<1, dim, Number>;

    DEAL_II_HOST_DEVICE
    PortableScalarField(FEEval *fe_eval, const unsigned int q)
      : fe_eval_(fe_eval)
      , q_(q)
    {}

    DEAL_II_HOST_DEVICE Number
    get_value() const
    {
      return fe_eval_->get_value(q_);
    }

    DEAL_II_HOST_DEVICE Gradient
    get_gradient() const
    {
      return fe_eval_->get_gradient(q_);
    }

    // Mirrors the intent of CPU FieldContainer::set_value_term().
    DEAL_II_HOST_DEVICE void
    set_value_term(const Number value) const
    {
      fe_eval_->submit_value(value, q_);
    }

    // Mirrors the intent of CPU FieldContainer::set_gradient_term().
    DEAL_II_HOST_DEVICE void
    set_gradient_term(const Gradient &gradient) const
    {
      fe_eval_->submit_gradient(gradient, q_);
    }

  private:
    FEEval       *fe_eval_;
    unsigned int  q_;
  };

  // PDE must expose static evaluation_flags / integration_flags and
  // a device-callable operator()(PortableScalarField&) const.
  template <int dim, int degree, typename Number, typename PDE>
  struct PortableScalarPDEQuadrature
  {
    using FEEval =
      dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, Number>;

    // Copy the runtime PDE parameters into the device functor.
    PDE pde{};

    DEAL_II_HOST_DEVICE void
    operator()(FEEval *fe_eval, const int q) const
    {
      PortableScalarField<dim, degree, Number> field(fe_eval, q);
      pde(field);
    }
  };

  template <int dim, int degree, typename Number, typename PDE>
  struct PortableScalarPDECell
  {
    using FEEval =
      dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, Number>;
    using MatrixFree = dealii::Portable::MatrixFree<dim, Number>;

    static constexpr unsigned int n_dofs_1d = degree + 1;
    static constexpr unsigned int n_local_dofs =
      dealii::Utilities::pow(degree + 1, dim);
    static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(degree + 1, dim);

    PDE pde{};

    DEAL_II_HOST_DEVICE void
    operator()(const unsigned int /*cell*/,
               const typename MatrixFree::Data *data,
               dealii::Portable::SharedData<dim, Number> *shared_data,
               const Number *src,
               Number *dst) const
    {
      FEEval fe_eval(data, shared_data);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(PDE::evaluation_flags);
      fe_eval.apply_for_each_quad_point(
        PortableScalarPDEQuadrature<dim, degree, Number, PDE>{pde});
      fe_eval.integrate(PDE::integration_flags);
      fe_eval.distribute_local_to_global(dst);
    }
  };
} // namespace prismspf
