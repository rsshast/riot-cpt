#ifndef CHARON_STOPPING_POWER_
#define CHARON_STOPPING_POWER_

#include <math.h>
#include <units.hpp>

namespace charon {

using Real = parthenon::Real;
using namespace units;

class MeanStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit MeanStoppingPower(Real val = Real(0.0)) : value_(val) {}

  KOKKOS_INLINE_FUNCTION
  MeanStoppingPower GetOnDevice() const { return *this; }

  // Constant mean dE/dx
  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx) const {
    Real E_loss = value_ * dx; // already multiplied by distance here.
    Real E_f = E_in - E_loss;
    return (E_f > EB) ? E_f : EB; // GPU compatible
  }

 private:
  Real value_;
};

class QuadStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit QuadStoppingPower(Real val = Real(0.0)) : value_(val) {}

  KOKKOS_INLINE_FUNCTION
  QuadStoppingPower GetOnDevice() const { return *this; }

  // Constant mean dE/dx
  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx, Real dt) const {
    Real dt2 = dt * dt;
    Real E_loss = value_ * dt2 * dx; // already multiplied by distance here. 
    Real E_f = E_in - E_loss;
    const Real edep = E_in - E_f;
    const Real max_edep = E_in - EB;
    return (edep < max_edep) ? edep : max_edep; // GPU compatible
  }

 private:
  Real value_;
};


class DensityLinearStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit DensityLinearStoppingPower(Real base = Real(0.0), Real rho_ref = Real(1.0),
                                      Real slope = Real(1.0))
      : base_(base), rho_ref_(rho_ref), slope_(slope) {}

  KOKKOS_INLINE_FUNCTION
  DensityLinearStoppingPower GetOnDevice() const { return *this; }

  KOKKOS_INLINE_FUNCTION
  Real Value(Real rho) const {
    const Real ref = (rho_ref_ > 0.0) ? rho_ref_ : 1.0;
    const Real factor = 1.0 + slope_ * (rho / ref - 1.0);
    return base_ * ((factor > 0.0) ? factor : Real(0.0));
  }

  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx, Real rho) const {
    const Real E_loss = Value(rho) * dx;
    const Real E_f = E_in - E_loss;
    return (E_f > EB) ? E_f : EB; // GPU compatible
  }

 private:
  Real base_;
  Real rho_ref_;
  Real slope_;
};

class TemperatureLinearStoppingPower {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit TemperatureLinearStoppingPower(Real base = Real(0.0),
                                          Real temperature_ref = Real(1.0),
                                          Real slope = Real(1.0))
      : base_(base), temperature_ref_(temperature_ref), slope_(slope) {}

  KOKKOS_INLINE_FUNCTION
  TemperatureLinearStoppingPower GetOnDevice() const { return *this; }

  KOKKOS_INLINE_FUNCTION
  Real Value(Real temperature) const {
    const Real ref = (temperature_ref_ > 0.0) ? temperature_ref_ : 1.0;
    const Real factor = slope_ * (ref / temperature);
    //const Real factor = 1.0 + slope_ * (temperature / ref - 1.0);
    return base_ * ((factor > 0.0) ? factor : Real(0.0));
  }

  KOKKOS_INLINE_FUNCTION
  Real ELoss(Real E_in, Real dx, Real temperature) const {
    const Real E_loss = Value(temperature) * dx;
    const Real E_f = E_in - E_loss;
    return (E_f > EB) ? E_f : EB; // GPU compatible
  }

 private:
  Real base_;
  Real temperature_ref_;
  Real slope_;
};


class LiPetrassoStoppingPower{
  public:
    KOKKOS_INLINE_FUNCTION
    // assumes that ne is already per volume element
    explicit LiPetrassoStoppingPower(
            int Zin=2,            // atomic number
            Real ne_0=1.,         // electron density
            Real na_0=1.,         // ion density
            Real Tb_upd=1.0       // electron temperature
            ) : Z(Zin), ne(ne_0), na(na_0), Tb(Tb_upd) {}

    KOKKOS_INLINE_FUNCTION
    LiPetrassoStoppingPower GetOnDevice() const { return *this; }

    // Temperatures
    KOKKOS_INLINE_FUNCTION
    Real T_alpha(Real E) const {return 2. * E / (3. * KB);}

    KOKKOS_INLINE_FUNCTION
    // Debye Length
    Real DebyeLength(Real E) const {
      Real inv_ld2 = (4.0 * M_PI / KB) * (
          (ne * Q * Q / Tb) +
          (na * (Z * Q) * (Z * Q) / T_alpha(E))
      );
      return std::sqrt(1. / inv_ld2);
    }

    KOKKOS_INLINE_FUNCTION
    // R perpendicular. Eqn 34 from LUMOS paper
    Real RPerp(Real E) const {
      Real m_ab = MALPHA * MBETA / (MALPHA + MBETA);
      Real u_bar = std::sqrt((2. * E)/ MALPHA);
      Real r1 = (Z * Q * Q) / (m_ab * u_bar * u_bar);
      Real r2 = HBAR / (2 * m_ab * u_bar);
      return std::max(r1,r2);
    }

    KOKKOS_INLINE_FUNCTION
    Real Lambda_ab(Real E) const {
      Real lambda_d = DebyeLength(E);
      Real r_perp   = RPerp(E);
      return log(lambda_d / r_perp);
    }

    KOKKOS_INLINE_FUNCTION
    Real VAlpha (Real E) const {
        return std::sqrt(2. * E  / MALPHA);
    }

    KOKKOS_INLINE_FUNCTION
    Real get_x(Real v) const {
        return (MBETA * v * v) / (2. * KB * Tb);
    }

    KOKKOS_INLINE_FUNCTION
    Real get_psi(Real x) const {
      if (x <= 0.0) return 0.0;
      return std::erf(std::sqrt(x)) - (2.0 / std::sqrt(M_PI)) * std::sqrt(x)* std::exp(-x);
    }

    KOKKOS_INLINE_FUNCTION
    Real get_psi_p(Real E, Real x) const {
      return 2. / std::sqrt(M_PI) * (std::sqrt(x) / std::exp(x));
    }

    KOKKOS_INLINE_FUNCTION
    Real get_G_params(Real l_ab, Real E, Real x) const {
      // Li-Petrasso correction factor on Trubnikov model
      Real m_ratio = MBETA / MALPHA;
      Real psi = get_psi(x);
      Real psi_p = get_psi_p(E, x);

      Real G = (psi - m_ratio * (psi_p - (1. / l_ab) * (psi + psi_p)));
      return G;
    }


    KOKKOS_INLINE_FUNCTION
    Real OmegaBeta2 () const { // Plamsa frequency squared
      return (4.0 * M_PI * ne * Q * Q) / (MBETA);
    }

    KOKKOS_INLINE_FUNCTION
    Real LP_Spower(Real E) const {
      if (E <= EB) return 0.0;
      Real l_ab = Lambda_ab(E);
      //if (l_ab < 2.0) l_ab = 2.0;
      Real v = VAlpha(E);
      Real x = get_x(v);
      Real Theta = 0.0;
      if (x > 1.0) { Theta = 1.0; } 

      Real wb2 = OmegaBeta2(); //electron plasma freq. 
      Real G = get_G_params(l_ab, E, x);
      Real coef = wb2 * (Z * Z) * (Q*Q) / (v * v);
      Real lp_spower =  coef * (G * (l_ab) + Theta * log(1.123 * std::sqrt(x)));
      //Real lp_spower =  coef * (G * std::log(l_ab) + Theta * log(1.123 * std::sqrt(x)));

      return lp_spower;
    }

    KOKKOS_INLINE_FUNCTION
    Real ELoss (Real Ei, Real spower, Real dx) const {
      Real Ef = Ei - spower * dx;
      return (Ef > EB) ? Ef : EB; // GPU compatible
    }

    KOKKOS_INLINE_FUNCTION
    Real Edep (Real Ei, Real Ef) const { //return std::min(Ei - Ef, Ei - EB); }
      const Real edep = Ei - Ef;
      const Real max_edep = Ei - EB;
      return (edep < max_edep) ? edep : max_edep;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Real BackgroundEnergy() { return 0.2 * MeV_to_ERG; }

  private:
    int Z;
    Real ne;
    Real na;
    Real Tb;
};

} // namespace charon

namespace mcblock {
using MeanStoppingPower = ::charon::MeanStoppingPower;
using DensityLinearStoppingPower = ::charon::DensityLinearStoppingPower;
using TemperatureLinearStoppingPower = ::charon::TemperatureLinearStoppingPower;
using LPStoppingPower   = ::charon::LiPetrassoStoppingPower;
} 
#endif
