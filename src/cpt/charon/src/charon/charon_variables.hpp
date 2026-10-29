//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// This file was made in part with generative AI.
#ifndef CHARON_CHARON_VARIABLES_HPP_
#define CHARON_CHARON_VARIABLES_HPP_

// C++ includes
#include <string>

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

//static const std::string photons_swarm_name = "photons";
static const std::string alphas_swarm_name = "alphas";

#define CHARON_FIELD_VARIABLE(ns, varname)                                             \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace field {
namespace charon {
CHARON_FIELD_VARIABLE(field.charon, energy_tally);
CHARON_FIELD_VARIABLE(field.charon, source_ew_per_cell);
CHARON_FIELD_VARIABLE(field.charon, source_num_per_cell);
CHARON_FIELD_VARIABLE(field.charon, delta_num_per_cell);
CHARON_FIELD_VARIABLE(field.charon, energy_delta);
CHARON_FIELD_VARIABLE(field.charon, mass_delta);
CHARON_FIELD_VARIABLE(field.charon, momentum_delta);
CHARON_FIELD_VARIABLE(field.charon, energy_tally_step);
CHARON_FIELD_VARIABLE(field.charon, deposited_energy_step);
CHARON_FIELD_VARIABLE(field.charon, energy_delta_step);
CHARON_FIELD_VARIABLE(field.charon, mass_delta_step);
CHARON_FIELD_VARIABLE(field.charon, momentum_delta_step);
CHARON_FIELD_VARIABLE(field.charon, active_ew_per_cell);
CHARON_FIELD_VARIABLE(field.charon, active_num_per_cell);
CHARON_FIELD_VARIABLE(field.charon, old_active_ew_per_cell); // for population control
CHARON_FIELD_VARIABLE(field.charon, active_ke_tally);
CHARON_FIELD_VARIABLE(field.charon, Phi);
CHARON_FIELD_VARIABLE(field.charon, Phi_uq);
CHARON_FIELD_VARIABLE(field.charon, ave_energy);
CHARON_FIELD_VARIABLE(field.charon, count_tally);
CHARON_FIELD_VARIABLE(field.charon, residence_tally);
CHARON_FIELD_VARIABLE(field.charon, spower_tally);
CHARON_FIELD_VARIABLE(field.charon, spower_sum);
CHARON_FIELD_VARIABLE(field.charon, spower_count);
// TN
CHARON_FIELD_VARIABLE(field.charon, plasma_velocity);
CHARON_FIELD_VARIABLE(field.charon, plasma_electron_density);
CHARON_FIELD_VARIABLE(field.charon, plasma_electron_temperature);
CHARON_FIELD_VARIABLE(field.charon, plasma_ion_temperature);
CHARON_FIELD_VARIABLE(field.charon, plasma_screening);
CHARON_FIELD_VARIABLE(field.charon, plasma_ion_density);
CHARON_FIELD_VARIABLE(field.charon, plasma_ion_charge);
CHARON_FIELD_VARIABLE(field.charon, pending_alphas);
CHARON_FIELD_VARIABLE(field.charon, physical_alphas);
CHARON_FIELD_VARIABLE(field.charon, sampled_alphas);
CHARON_FIELD_VARIABLE(field.charon, electron_heat_step);
CHARON_FIELD_VARIABLE(field.charon, ion_heat_step);
CHARON_FIELD_VARIABLE(field.charon, electron_heat);
CHARON_FIELD_VARIABLE(field.charon, ion_heat);
CHARON_FIELD_VARIABLE(field.charon, escaped_alphas);
CHARON_FIELD_VARIABLE(field.charon, escaped_energy);

namespace host {
typedef HOST_DENSITY density;
typedef HOST_TEMPERATURE temperature;
typedef HOST_ALPHA_SOURCE alpha_source;
} // namespace host
} // namespace charon
} // namespace field

namespace particle {
//static const std::string alphas_swarm_name = "alphas";
namespace alphas {
#define CHARON_SWARM_VARIABLE(type, ns, varname)                                      \
  struct varname : public parthenon::swarm_variable_names::base_t<type> {               \
    template <class... Ts>                                                              \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                        \
        : parthenon::swarm_variable_names::base_t<type>(std::forward<Ts>(args)...) {}   \
    static std::string name() { return #ns "." #varname; }                              \
  }

CHARON_SWARM_VARIABLE(Real, particle.alphas, time);
CHARON_SWARM_VARIABLE(Real, particle.alphas, weight);
CHARON_SWARM_VARIABLE(Real, particle.alphas, energy);
CHARON_SWARM_VARIABLE(Real, particle.alphas, v);
CHARON_SWARM_VARIABLE(int, particle.alphas, ijk);

#undef CHARON_SWARM_VARIABLE
} // namespace alphas
} // namespace particle

#endif // CHARON_CHARON_VARIABLES_HPP_
