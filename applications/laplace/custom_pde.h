// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/core/pde_operator_base.h>

#ifdef PRISMS_PF_GPU
#  include <prismspf/solvers/portable_scalar_pde_operator.h>
#endif

#include <random>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

#ifdef PRISMS_PF_GPU
  /**
   * @brief Compile-time GPU counterpart of compute_lhs() for Laplace solve block 1.
   *
   * No host-side FieldContainer or virtual PDE callback is used on the GPU.
   * This is a scalar, single-field example, not general multi-field support.
   */
  struct GpuLHS
  {
    static constexpr dealii::EvaluationFlags::EvaluationFlags evaluation_flags =
      dealii::EvaluationFlags::gradients;
    static constexpr dealii::EvaluationFlags::EvaluationFlags integration_flags =
      dealii::EvaluationFlags::gradients;

    template <typename ScalarField>
    DEAL_II_HOST_DEVICE void
    operator()(const ScalarField &field) const
    {
      field.set_gradient_term(field.get_gradient());
    }
  };

  using DeviceVector = typename PDEOperatorBase<dim, degree, number>::DeviceVector;

  bool
  supports_gpu_lhs(const unsigned int solve_block_id) const override
  {
    return solve_block_id == 1;
  }

  bool
  compute_gpu_lhs(const unsigned int solve_block_id,
                  const unsigned int field_index,
                  const MatrixFreeManager<dim, number> &manager,
                  DeviceVector &dst,
                  const DeviceVector &src) const override
  {
    if (solve_block_id != 1 || field_index != 0)
      return false;

    PortableScalarPDEOperator<dim, degree, number, GpuLHS> op(manager, field_index);
    // Match CPU MFOperator::compute_operator(): cell_loop alone does not
    // copy constrained src values into dst. This avoids adding identity
    // rows during the CPU-CG -> GPU-matvec bridge.
    dst = number(0);
    op.apply_cells(dst, src);
    return true;
  }

#endif

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
  {}

  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] const SimulationTimer    &sim_timer,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number &vector_component_value) const override
  {
    [[maybe_unused]] const double x = (dim > 0) ? point[0] : 0.0;
    [[maybe_unused]] const double y = (dim > 1) ? point[1] : 0.0;
    [[maybe_unused]] const double z = (dim > 2) ? point[2] : 0.0;

    if (boundary_id == RectangularMesh<dim>::Boundary::Left)
      {
        using std::cos;
        scalar_value = 50.0 * cos(x / 4.0) * cos(y / 4.0);
        return;
      }
    if (boundary_id == RectangularMesh<dim>::Boundary::Right)
      {
        scalar_value = y / 2.0;
        return;
      }
    if (boundary_id == RectangularMesh<dim>::Boundary::Bottom)
      {
        scalar_value = x * (100.0 - x) / 50.0;
        return;
      }
    if (boundary_id == RectangularMesh<dim>::Boundary::Top)
      {
        scalar_value = 0.0;
        return;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 1) // linear rhs
      {
        variable_list.set_value_term(0, ScalarValue(0.0));
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 1) // linear lhs
      {
        ScalarGrad ux = variable_list.template get_gradient<Scalar, LHS>(0);
        variable_list.set_gradient_term(0, ux);
      }
  }
};

PRISMS_PF_END_NAMESPACE
