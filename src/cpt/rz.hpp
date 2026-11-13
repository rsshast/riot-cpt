// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.
#ifndef CPT_RZ_HPP_
#define CPT_RZ_HPP_
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
namespace CPT::rz {
void Initialize(parthenon::ParameterInput *,parthenon::StateDescriptor *);
void Step(parthenon::Mesh *,parthenon::Real,parthenon::Real,int);
}
#endif
