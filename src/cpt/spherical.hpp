// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.
#ifndef CPT_SPHERICAL_HPP_
#define CPT_SPHERICAL_HPP_

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace CPT::spherical {
// Rank-owned Cartesian histories; no Parthenon swarm is registered or accessed.
void Initialize(parthenon::ParameterInput *pin, parthenon::StateDescriptor *pkg);
void Step(parthenon::Mesh *pm, parthenon::Real start, parthenon::Real dt, int cycle);
// Restore extensive source/accounting fields after a leaf-mesh change, before burn
// or checkpoint output. Particle trajectories and rank ownership are unchanged.
void RestoreInventories(parthenon::Mesh *pm);
} // namespace CPT::spherical
#endif
