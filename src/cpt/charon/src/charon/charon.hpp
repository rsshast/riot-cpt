//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef CHARON_CHARON_HPP_
#define CHARON_CHARON_HPP_

// C/C++ includes
#include <limits>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cmath>

// Kokkos includes
#include "Kokkos_Random.hpp"
// NOTE(): these typedefs must appear before Charon includes for compilation
typedef Kokkos::Random_XorShift64_Pool<> RngPool;
typedef RngPool::generator_type RngGen;

// Parthenon includes
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon;
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Host configuration file
#include "charon_config.hpp"

// Charon includes
#include "boundaries.hpp"
#include "charon_variables.hpp"
#include "stopping_power.hpp" 

namespace charon {

std::shared_ptr<parthenon::StateDescriptor>
Initialize(parthenon::ParameterInput *pin, std::string block_name = "charon",
           bool allow_compact_tallies = false);

// Model enums
enum class SourceStrategy {
  t0, isotropic, center, point, xhat, line, plane, fuel, qfield
};
enum class ParticleType {alphas};

// Initialization nulls
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto CharonNull() {
  return std::numeric_limits<T>::quiet_NaN();
}
template <>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto CharonNull<int>() {
  return std::numeric_limits<int>::max();
}

// Tasks
TaskStatus TransportAlphas(MeshData<Real> *md, const Real t_start, const Real dt);
TaskStatus CheckCompletion(MeshData<Real> *md, const Real t_end);

template <typename T>
TaskStatus SourceAlphas(T *md, const Real t_start, const Real dt);

TaskStatus DefragParticles(MeshData<Real> *md);
TaskStatus UpdateDerivedTransportFields(MeshData<Real> *md, const Real dt);

template <typename T>
TaskStatus EvaluateRadiationEnergy(T *md);

TaskStatus ControlPopulation(MeshData<Real> *md);
void PrepareTransportOutput(Mesh *pmesh, ParameterInput *pin, SimTime &tm);
void ResetTransportOutputTallies(Mesh *pmesh, ParameterInput *pin, SimTime &tm);
void ResetTransportOutputTalliesIfOutputAdvanced(Mesh *pmesh, ParameterInput *pin,
                                                 SimTime &tm);

// TaskCollection for radiation step
TaskCollection RadiationStep(Mesh *pmesh, const Real t_start, const Real dt);

// Functions
Real EstimateTimestepMesh(MeshData<Real> *md);
void InitializeRadiation(MeshBlockData<Real> *mbd);
inline void InitializeRadiation(MeshBlockData<Real> *mbd, bool) {
  InitializeRadiation(mbd);
}

} // namespace charon

#endif // CHARON_CHARON_HPP_
