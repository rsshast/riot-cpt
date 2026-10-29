//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//========================================================================================
// This file was made in part with generative AI.
#ifndef CPT_CPT_HPP_
#define CPT_CPT_HPP_

#include <memory>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace CPT {

std::shared_ptr<parthenon::StateDescriptor>
Initialize(parthenon::ParameterInput *pin,
           parthenon::StateDescriptor *materials);

parthenon::TaskListStatus Step(parthenon::Mesh *pm, parthenon::SimTime &tm,
                               const parthenon::Real dt);

} // namespace CPT

#endif // CPT_CPT_HPP_
