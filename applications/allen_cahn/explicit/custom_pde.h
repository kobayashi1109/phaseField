// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#ifdef PRISMS_PF_GPU
#  include <prismspf/solvers/portable_scalar_pde_operator.h>
#endif

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

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , m_well(get_user_inputs().user_constants.get_double("m_well"))
    , kappa(get_user_inputs().user_constants.get_double("kappa"))
  {}

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;

    constexpr number center[12][3] = {
      {0.1, 0.3,  0},
      {0.8, 0.7,  0},
      {0.5, 0.2,  0},
      {0.4, 0.4,  0},
      {0.3, 0.9,  0},
      {0.8, 0.1,  0},
      {0.9, 0.5,  0},
      {0.0, 0.1,  0},
      {0.1, 0.6,  0},
      {0.5, 0.6,  0},
      {1,   1,    0},
      {0.7, 0.95, 0}
    };
    constexpr number rad[12] = {12, 14, 19, 16, 11, 12, 17, 15, 20, 10, 11, 14};
    number           dist    = 0.0;
    for (unsigned int i = 0; i < 12; i++)
      {
        dist = 0.0;
        for (unsigned int dir = 0; dir < dim; dir++)
          {
            const number comp_diff = point[dir] - center[i][dir] * mesh_size[dir];
            dist += comp_diff * comp_diff;
          }
        dist = std::sqrt(dist);

        scalar_value += 0.5 * (1.0 - std::tanh((dist - rad[i]) / 1.5));
      }
    scalar_value = std::min(scalar_value, number(1.0));
  }

  void
  compute_rhs(FieldContainer<dim, degree, number> &variable_list,
              const SimulationTimer               &sim_timer,
              unsigned int                         solve_block_id) const override
  {
    if (solve_block_id == 1) // explicit n
      {
        ScalarValue n_val  = variable_list.template get_value<Scalar, OldOne>(0);
        ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, OldOne>(0);
        ScalarValue f_well = 4.0 * n_val * (n_val - 1.0) * (n_val - 0.5);
        ScalarValue eq_n   = n_val - sim_timer.get_timestep() * m_well * f_well;
        ScalarGrad  eqx_n  = -sim_timer.get_timestep() * kappa * m_well * n_grad;

        variable_list.set_value_term(0, eq_n);
        variable_list.set_gradient_term(0, eqx_n);
      }
    else if (solve_block_id == 2) // postprocess
      {
        ScalarValue n_val  = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        ScalarValue f_tot  = 0.0;
        ScalarValue f_chem =
          n_val * n_val * n_val * n_val - 2.0 * n_val * n_val * n_val + n_val * n_val;
        ScalarValue f_grad = 0.5 * kappa * n_grad.norm_square();
        f_tot              = f_chem + f_grad;

        variable_list.set_value_term(1, n_grad.norm());
        variable_list.set_value_term(2, f_tot);
      }
  }

#ifdef PRISMS_PF_GPU
  using DeviceVector =
    typename PDEOperatorBase<dim, degree, number>::DeviceVector;

public:
  // Public for CUDA/Kokkos-generated kernel registration code.
  // Device-callable version of the CPU RHS for the primary explicit field.
  // The input represents old_1(n). Inverse mass scaling is done by the solver.
  struct GpuRHS
  {
    number dt;
    number mobility;
    number gradient_energy;

    // deal.II 9.6's EvaluationFlags::operator| is not constexpr.
    // Combine the enum bits using the built-in integral operator instead.
    static constexpr auto evaluation_flags =
      static_cast<decltype(dealii::EvaluationFlags::values)>(
        static_cast<unsigned int>(dealii::EvaluationFlags::values) |
        static_cast<unsigned int>(dealii::EvaluationFlags::gradients));

    static constexpr auto integration_flags = evaluation_flags;

    DEAL_II_HOST_DEVICE void
    operator()(PortableScalarField<dim, degree, number> &field) const
    {
      const number n_val = field.get_value();
      const auto n_grad = field.get_gradient();

      const number f_well =
        number(4.0) * n_val * (n_val - number(1.0)) *
        (n_val - number(0.5));

      const number eq_n = n_val - dt * mobility * f_well;
      const auto eqx_n = -dt * gradient_energy * mobility * n_grad;

      field.set_value_term(eq_n);
      field.set_gradient_term(eqx_n);
    }
  };

private:
  bool
  supports_gpu_rhs(unsigned int solve_block_id) const override
  {
    return solve_block_id == 1;
  }

  // This example has one self-dependent primary scalar and read-only
  // postprocessing of its Current value. It has no pre/post solve hooks.
  bool
  supports_gpu_resident_explicit(unsigned int solve_block_id) const override
  {
    return solve_block_id == 1;
  }

  bool
  compute_gpu_rhs(
    unsigned int solve_block_id,
    unsigned int field_index,
    const MatrixFreeManager<dim, number> &manager,
    const SimulationTimer &sim_timer,
    DeviceVector &dst,
    const DeviceVector &src) const override
  {
    if (solve_block_id != 1 || field_index != 0)
      return false;

    PortableScalarPDEOperator<dim, degree, number, GpuRHS>
      gpu_operator(manager, field_index);

    const GpuRHS pde{
      static_cast<number>(sim_timer.get_timestep()),
      m_well,
      kappa
    };

    // Assemble only the RHS. Unlike vmult(), apply_cells() does not impose
    // the LHS identity-row treatment for constrained DoFs.
    dst = number(0);
    gpu_operator.apply_cells(dst, src, pde);

    return true;
  }
#endif

  number m_well;
  number kappa;
};

PRISMS_PF_END_NAMESPACE
