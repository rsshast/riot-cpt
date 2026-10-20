#ifndef CHARON_UNITS_HPP_
#define CHARON_UNITS_HPP_

#include <parthenon/package.hpp>
#include <utils/robust.hpp>

namespace charon {
namespace units {

using Real = parthenon::Real;

// Physics Constants
// no need for EPS0 with cgs units
static constexpr Real HBAR = 1.0546e-27;                // cm^2g/s
static constexpr Real KB = 1.380649e-16;                //Boltzmann constant, cm^2g/(s^2k)

// Mass
static constexpr Real MALPHA = 6.644657e-24;            // g
static constexpr Real MBETA = 9.109383e-28;             // g
static constexpr Real NAVO = 6.022e23;                  // atoms per mole
static constexpr Real MPROTON = 1.6605390666e-24;       // g

// Charge
static constexpr Real Q = 4.803204e-10;                 // fundamental charge, Coulombs
static constexpr int Z = 2;                             // alpha particle atomic number
static constexpr int Ze = 1;                            // electron particle atomic number
static constexpr int ABAR = 1;                          // average atomic number of DT plasma

// Energy
static constexpr Real MeV_to_ERG = 1.60218e-6;
static constexpr Real ERG = 1.0e7;                      // conversion from MeV to Erg
static constexpr Real E0 = 3.54 * MeV_to_ERG;           // sourcing energy
//static constexpr Real EB = 0.0  * MeV_to_ERG;           // background energy
static constexpr Real EB = 0.2  * MeV_to_ERG;           // background energy

// set tolerance for checking particle 0-velocity (i.e. if from DDMC block)
constexpr Real EPS = parthenon::robust::EPS();

// --- Helper Functions ---

} // namespace units
} // namespace charon

#endif // CHARON_UNITS_HPP_
