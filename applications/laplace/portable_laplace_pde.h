#pragma once

// Bind the device-callable PDE declared in CustomPDE to the generic scalar
// matrix-free GPU cell operator. The physical term is defined in custom_pde.h.

#include "custom_pde.h"

#include <prismspf/solvers/portable_scalar_pde.h>

namespace prismspf
{
  template <int dim, int degree, typename Number>
  using PortableLaplacePDECell =
    PortableScalarPDECell<dim,
                          degree,
                          Number,
                          typename CustomPDE<dim, degree, Number>::GpuLHS>;
} // namespace prismspf
