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
// Discrete Diffusion Monte Carlo
// C/C++ includes
#include <algorithm>
#include <chrono>
#include <fstream>

// Parthenon includes
#include <utils/robust.hpp>

// Charon includes
#include "charon.hpp"
#include "charon_utils.hpp"
#include "charon_variables.hpp"
#include "stopping_power.hpp"
#include "transport_utils.hpp"
#include "units.hpp"

namespace charon {

// One particle lifecycle for both material adapters. Compile-time selection keeps
// plasma-only fields out of legacy packs and preserves the legacy segment pusher.
namespace {
namespace fj = field::charon;
namespace fjh = field::charon::host;
namespace pa = particle::alphas;
namespace sp = swarm_position;

template <bool Diagnostics, bool Compact = false, bool Plasma = false>
TaskStatus TransportAlphasImpl(MeshData<Real> *md, const Real t_start, const Real dt) {
  auto pm = md->GetParentPointer();
  auto &cpt_pkg = pm->packages.Get("charon");
  bool has_particles = false;
  for (int b = 0; b < md->NumBlocks(); ++b)
    has_particles =
        has_particles || md->GetSwarmData(b)->Get(alphas_swarm_name)->GetNumActive() > 0;
  if (!has_particles) return TaskStatus::complete;

  const bool time_transport = cpt_pkg->Param<bool>("weak_scaling_diagnostics");
  std::chrono::steady_clock::time_point transport_wall_start;
  if (time_transport) {
    Kokkos::fence();
    transport_wall_start = std::chrono::steady_clock::now();
  }
  auto &rng_pool = cpt_pkg->Param<RngPool>("rng_pool");
  const bool tally_mass_momentum = cpt_pkg->Param<bool>("tally_mass_momentum");
  const bool verbose = cpt_pkg->Param<bool>("verbose");
  const auto model = cpt_pkg->Param<std::string>("stopping_power_model");
  const bool use_lp = model == "LiPetrasso", lp = use_lp;
  const bool use_quad = model == "quadratic";
  const bool use_density_linear = model == "density_linear";
  const bool use_temperature_linear = model == "temperature_linear";
  const Real stopping_power_value = cpt_pkg->Param<Real>("stopping_power_value");
  const Real constant = stopping_power_value;
  const Real stopping_power_density_reference =
      cpt_pkg->Param<Real>("stopping_power_density_reference");
  const Real stopping_power_density_slope =
      cpt_pkg->Param<Real>("stopping_power_density_slope");
  const Real stopping_power_temperature_reference =
      cpt_pkg->Param<Real>("stopping_power_temperature_reference");
  const Real stopping_power_temperature_slope =
      cpt_pkg->Param<Real>("stopping_power_temperature_slope");
  Real cutoff = units::EB, ion_fraction = 0, tolerance = 0, fraction = 0;
  int ns = 0, max_steps = 0;
  ParArray1D<Real> masses;
  if constexpr (Plasma) {
    PARTHENON_REQUIRE(pm->ndim == 3, "TN plasma transport requires a 3-D Cartesian mesh");
    cutoff = cpt_pkg->Param<Real>("plasma_thermalization_energy");
    ion_fraction = cpt_pkg->Param<Real>("constant_ion_fraction");
    tolerance = cpt_pkg->Param<Real>("plasma_velocity_tolerance");
    fraction = cpt_pkg->Param<Real>("plasma_fractional_step");
    max_steps = cpt_pkg->Param<int>("plasma_max_substeps");
    ns = cpt_pkg->Param<int>("plasma_num_species");
    masses = cpt_pkg->Param<ParArray1D<Real>>("plasma_species_mass");
  }

  auto desc = [&]() {
    if constexpr (Plasma) {
      return MakePackDescriptor<
          fj::plasma_velocity, fj::plasma_electron_density,
          fj::plasma_electron_temperature, fj::plasma_ion_temperature,
          fj::plasma_screening, fj::plasma_ion_density, fj::plasma_ion_charge,
          fj::electron_heat_step, fj::ion_heat_step, fj::electron_heat, fj::ion_heat,
          fj::escaped_alphas, fj::escaped_energy>(pm->resolved_packages.get());
    } else if constexpr (Compact) {
      return MakePackDescriptor<fjh::density, fjh::temperature>(
          pm->resolved_packages.get());
    } else if constexpr (Diagnostics) {
      return MakePackDescriptor<fjh::density, fjh::temperature, fj::energy_delta_step,
                                fj::mass_delta_step, fj::momentum_delta_step, fj::Phi,
                                fj::Phi_uq, fj::ave_energy, fj::count_tally,
                                fj::residence_tally>(pm->resolved_packages.get());
    } else {
      return MakePackDescriptor<fjh::density, fjh::temperature, fj::energy_delta_step,
                                fj::mass_delta_step, fj::momentum_delta_step>(
          pm->resolved_packages.get());
    }
  }();
  auto vmesh = desc.GetPack(md);
  auto heat_desc = [&]() {
    if constexpr (Compact)
      return MakePackDescriptor<fj::deposited_energy_step>(pm->resolved_packages.get());
    else
      return MakePackDescriptor<fj::energy_tally_step>(pm->resolved_packages.get());
  }();
  auto heat = heat_desc.GetPack(md);
  auto diag_desc = [&]() {
    if constexpr (Diagnostics)
      return MakePackDescriptor<fj::spower_sum, fj::spower_count>(
          pm->resolved_packages.get());
    else if constexpr (Plasma)
      return MakePackDescriptor<fj::plasma_screening>(pm->resolved_packages.get());
    else
      return MakePackDescriptor<fjh::density>(pm->resolved_packages.get());
  }();
  auto diag = diag_desc.GetPack(md);
  auto pdesc_r = MakeSwarmPackDescriptor<sp::x, sp::y, sp::z, pa::v, pa::energy,
                                         pa::weight, pa::time>(alphas_swarm_name);
  auto pdesc_i = MakeSwarmPackDescriptor<pa::ijk>(alphas_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);
  const bool multi_d = pm->ndim >= 2, three_d = pm->ndim == 3;
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);
  const Real end = t_start + dt;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "TransportCPT", DevExecSpace(), 0, ppack_r.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        auto swarm_d = ppack_r.GetContext(b);
        if (!swarm_d.IsActive(n)) return;
        Real &x = ppack_r(b, sp::x(), n), &y = ppack_r(b, sp::y(), n),
             &z = ppack_r(b, sp::z(), n);
        Real &t = ppack_r(b, pa::time(), n);
        Real &vx = ppack_r(b, pa::v(0), n), &vy = ppack_r(b, pa::v(1), n),
             &vz = ppack_r(b, pa::v(2), n);
        Real &ee = ppack_r(b, pa::energy(), n);
        const Real ww = ppack_r(b, pa::weight(), n);
        int &ip = ppack_i(b, pa::ijk(0), n), &jp = ppack_i(b, pa::ijk(1), n),
            &kp = ppack_i(b, pa::ijk(2), n);
        const auto outside_global = [&]() {
          return x < swarm_d.x_min_global_ || x >= swarm_d.x_max_global_ ||
                 (multi_d && (y < swarm_d.y_min_global_ || y >= swarm_d.y_max_global_)) ||
                 (three_d && (z < swarm_d.z_min_global_ || z >= swarm_d.z_max_global_));
        };
        if (outside_global()) {
          swarm_d.MarkParticleForRemoval(n);
          return;
        }
        swarm_d.Xtoijk(x, y, z, ip, jp, kp);
#ifndef KOKKOS_ENABLE_CUDA
        if (verbose) {
          // Log this particle in its owning block only.
          const auto mode = t_start == 0.0 ? std::ios::out | std::ios::trunc
                                           : std::ios::out | std::ios::app;
          std::ofstream file("particle_history_rank" +
                                 std::to_string(parthenon::Globals::my_rank) + ".txt",
                             mode);
          file << b << " " << n << " " << x << " " << y << " " << z << " " << t << " "
               << ee << "\n";
        }
#endif
        auto &coords = vmesh.GetCoordinates(b);
        Real phi_local = 0, e_density_accum = 0, residence_density_accum = 0;
        int ip_p = ip, jp_p = jp, kp_p = kp;
        bool on_block = true;
        int steps = 0;
        while (on_block && t < end) {
          if constexpr (Plasma)
            PARTHENON_REQUIRE(
                ++steps <= max_steps,
                "Plasma substep limit exceeded: inspect LP state/tolerances");
          PARTHENON_REQUIRE(ip >= ib.s && ip <= ib.e && jp >= jb.s && jp <= jb.e &&
                                kp >= kb.s && kp <= kb.e,
                            "Particle outside owning interior cell");
          // Keep the traversed cell until all segment and thermalization tallies
          // are complete; migration may change the particle's logical indices.
          const Real t_prev = t, ee_i = ee;
          const Real dv = coords.CellVolume(kp, jp, ip);
          Real segment_distance = 0, segment_stopping = 0, E_dep = 0;
          Real qe = 0, qi = 0;
          Real old_velocity[3] = {0, 0, 0};
          if constexpr (!Plasma && !Compact) {
            if (tally_mass_momentum)
              for (int d = 0; d < 3; ++d)
                old_velocity[d] = ppack_r(b, pa::v(d), n);
          }
          bool thermal = false;
          if constexpr (Plasma) {
            Real u[3], direction[3], position[3] = {x, y, z};
            Real relative2 = 0;
            for (int d = 0; d < 3; ++d) {
              u[d] = vmesh(b, fj::plasma_velocity(d), kp, jp, ip);
              direction[d] = ppack_r(b, pa::v(d), n) - u[d];
              relative2 += direction[d] * direction[d];
            }
            const Real energy = 0.5 * units::MALPHA * relative2;
            PARTHENON_REQUIRE(std::isfinite(energy), "Invalid relative alpha energy");
            const bool interacting =
                vmesh(b, fj::plasma_screening(), kp, jp, ip) > 0 || (!lp && constant > 0);
            thermal = interacting && energy <= cutoff * (1 + 1.e-8);
            if (thermal) {
              // Unresolved thermalization tail goes to ions, including particles that
              // enter a new fluid frame already below the cutoff.
              qi = energy;
            } else {
              const Real speed = std::sqrt(relative2);
              if (speed == 0 && !interacting) {
                t = end;
                break;
              }
              for (int d = 0; d < 3; ++d)
                direction[d] /= speed;
              
              const auto stopping = [&](Real e, Real &se, Real &si) {
                if (!lp) {
                  se = constant * (1 - ion_fraction);
                  si = constant * ion_fraction;
                  return;
                }
                const Real screening = vmesh(b, fj::plasma_screening(), kp, jp, ip);
                se = PlasmaStopping::Channel(
                    e, vmesh(b, fj::plasma_electron_density(), kp, jp, ip), units::MBETA,
                    1, vmesh(b, fj::plasma_electron_temperature(), kp, jp, ip),
                    screening);
                si = 0;
                for (int s = 0; s < ns; ++s)
                  si += PlasmaStopping::Channel(
                      e, vmesh(b, fj::plasma_ion_density(s), kp, jp, ip), masses(s),
                      vmesh(b, fj::plasma_ion_charge(s), kp, jp, ip),
                      vmesh(b, fj::plasma_ion_temperature(), kp, jp, ip), screening);
              };
/*const auto stopping = [&](Real e, Real &se, Real &si) {
  if (!lp) {
    se = constant * (1 - ion_fraction);
    si = constant * ion_fraction;
    return;
  }

  const Real screening = vmesh(b, fj::plasma_screening(), kp, jp, ip);
  const Real ne = vmesh(b, fj::plasma_electron_density(), kp, jp, ip);
  const Real te = vmesh(b, fj::plasma_electron_temperature(), kp, jp, ip);
  const Real ti = vmesh(b, fj::plasma_ion_temperature(), kp, jp, ip);

  const auto checked_channel =
      [&](int species, Real density, Real mass, Real charge,
          Real temperature) -> Real {
    // species=-1 identifies electrons; nonnegative values index ion channels.
    const auto positive_finite = [](Real value) {
      return std::isfinite(value) && value > 0;
    };

    const bool valid_inputs =
        positive_finite(e) &&
        std::isfinite(density) && density >= 0 &&
        positive_finite(mass) &&
        std::isfinite(charge) &&
        positive_finite(temperature) &&
        positive_finite(screening);

    // Match Channel's inactive-species behavior, but detect malformed inputs
    // rather than silently skipping NaN/negative density or invalid charge.
    if (std::isfinite(density) && density >= 0 &&
        std::isfinite(charge) && (density == 0 || charge == 0))
      return 0;

    // Sentinels mean the calculation was skipped because inputs were invalid.
    Real v2 = -1, relative2 = -1, reduced_mass = -1;
    Real classical = -1, quantum = -1, rmin = -1;
    Real scale = -1, argument = -1, log_lambda = -1;

    if (valid_inputs) {
      v2 = 2.0 * e / units::MALPHA;
      relative2 = v2 + 3.0 * units::KB * temperature / mass;
      reduced_mass = units::MALPHA * mass / (units::MALPHA + mass);

      if (positive_finite(relative2) && positive_finite(reduced_mass)) {
        classical = std::abs(2.0 * charge) * units::Q * units::Q /
                    (reduced_mass * relative2);
        quantum = units::HBAR /
                  (2.0 * reduced_mass * std::sqrt(relative2));
        rmin = std::max(classical, quantum);
        scale = std::sqrt(screening) * rmin;

        if (positive_finite(scale)) {
          argument = 1.0 / scale;
          if (positive_finite(argument))
            log_lambda = std::log(argument);
        }
      }
    }

    if (!valid_inputs || !positive_finite(log_lambda)) {
      // One record contains enough information to associate the failure with
      // a particle, a cell, and either the electron or a particular ion channel.
      printf(
          "\n[LP_BAD] b=%d particle=%d cell=(%d,%d,%d) species=%d "
          "t=%.17e xyz=(%.17e,%.17e,%.17e) "
          "E_eval_erg=%.17e E_eval_MeV=%.17e E_segment_erg=%.17e "
          "n=%.17e mass_g=%.17e Z=%.17e T_K=%.17e "
          "ne=%.17e Te_K=%.17e Ti_K=%.17e inv_debye2=%.17e "
          "v2=%.17e relative2=%.17e reduced_mass_g=%.17e "
          "b_classical_cm=%.17e b_quantum_cm=%.17e rmin_cm=%.17e "
          "sqrt_screening_times_rmin=%.17e "
          "lambda_argument=%.17e log_lambda=%.17e\n",
          b, n, kp, jp, ip, species,
          double(t), double(x), double(y), double(z),
          double(e), double(e / units::MeV_to_ERG), double(energy),
          double(density), double(mass), double(charge), double(temperature),
          double(ne), double(te), double(ti), double(screening),
          double(v2), double(relative2), double(reduced_mass),
          double(classical), double(quantum), double(rmin),
          double(scale), double(argument), double(log_lambda));

      // Dump all species to expose which ones contribute to screening.
      for (int s = 0; s < ns; ++s) {
        printf("[LP_ION] b=%d particle=%d species=%d "
               "mass_g=%.17e n=%.17e Z=%.17e Ti_K=%.17e\n",
               b, n, s, double(masses(s)),
               double(vmesh(b, fj::plasma_ion_density(s), kp, jp, ip)),
               double(vmesh(b, fj::plasma_ion_charge(s), kp, jp, ip)),
               double(ti));
      }

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
      fflush(stdout);
#endif
      PARTHENON_FAIL("LP diagnostic: invalid inputs or Coulomb logarithm");
      return 0;
    }

    return PlasmaStopping::Channel(
        e, density, mass, charge, temperature, screening);
  };

  se = checked_channel(-1, ne, units::MBETA, 1.0, te);
  si = 0;
  for (int s = 0; s < ns; ++s) {
    si += checked_channel(
        s, vmesh(b, fj::plasma_ion_density(s), kp, jp, ip), masses(s),
        vmesh(b, fj::plasma_ion_charge(s), kp, jp, ip), ti);
  }
};*/
              Real se, si;
              stopping(energy, se, si);
              const Real total = se + si;
              Real h = end - t;
              if (total > 0) {
                h = std::min(h, fraction * units::MALPHA * speed / total);
                h = std::min(h, (energy - cutoff) / (total * speed));
              }
              const Real width[3] = {coords.template Dxc<X1DIR>(kp, jp, ip),
                                     coords.template Dxc<X2DIR>(kp, jp, ip),
                                     coords.template Dxc<X3DIR>(kp, jp, ip)};
              const Real center[3] = {coords.template Xc<X1DIR>(ip),
                                      coords.template Xc<X2DIR>(jp),
                                      coords.template Xc<X3DIR>(kp)};
              // Bound each component throughout drag toward u, including reversals.
              for (int d = 0; d < 3; ++d) {
                const Real lab = ppack_r(b, pa::v(d), n);
                const Real positive = std::max(Real(0), std::max(lab, u[d]));
                const Real negative = std::max(Real(0), std::max(-lab, -u[d]));
                if (positive > 0)
                  h = std::min(h, (center[d] + 0.5 * width[d] - position[d]) / positive);
                if (negative > 0)
                  h = std::min(h, (position[d] - center[d] + 0.5 * width[d]) / negative);
              }
              PARTHENON_REQUIRE(std::isfinite(h) && h > 0 && t + h > t,
                                "Plasma transport timestep underflow");
              const auto step = TakeDragStep(energy, interacting ? cutoff : Real(0), h,
                                             tolerance, stopping);
              h = step.dt;
              const Real final_speed = step.speed, mid_e = step.electron_stopping,
                         mid_i = step.ion_stopping;
              PARTHENON_REQUIRE(t + h > t, "LP integration timestep underflow");
              const Real mid_total = mid_e + mid_i;
              const Real loss =
                  mid_total > 0
                      ? std::max(Real(0),
                                 energy - 0.5 * units::MALPHA * final_speed * final_speed)
                      : Real(0);
              qe = mid_total > 0 ? loss * mid_e / mid_total : 0;
              qi = loss - qe;
              for (int d = 0; d < 3; ++d) {
                position[d] += (u[d] + direction[d] * 0.5 * (speed + final_speed)) * h;
                ppack_r(b, pa::v(d), n) = u[d] + direction[d] * final_speed;
              }
              t += h;
              segment_distance = 0.5 * (speed + final_speed) * h;
              thermal = interacting && 0.5 * units::MALPHA * final_speed * final_speed <=
                                           cutoff * (1 + 1.e-8);
              if (thermal) qi += 0.5 * units::MALPHA * final_speed * final_speed;
              if constexpr (Diagnostics) {
                Kokkos::atomic_add(&diag(b, fj::spower_sum(), kp, jp, ip), mid_total);
                Kokkos::atomic_add(&diag(b, fj::spower_count(), kp, jp, ip), Real(1));
              }
              // Resolve face roundoff by a tiny position nudge only; inventory remains
              // attributed to the cell traversed by the accepted segment.
              if (!thermal)
                for (int d = 0; d < 3; ++d) {
                  const Real epsilon = 1.e-10 * width[d];
                  if (ppack_r(b, pa::v(d), n) > 0 &&
                      center[d] + 0.5 * width[d] - position[d] < epsilon)
                    position[d] = center[d] + 0.5 * width[d] + epsilon;
                  if (ppack_r(b, pa::v(d), n) < 0 &&
                      position[d] - (center[d] - 0.5 * width[d]) < epsilon)
                    position[d] = center[d] - 0.5 * width[d] - epsilon;
                }
              x = position[0];
              y = position[1];
              z = position[2];
            }

          } else {
            const Real dx_i = coords.template Dxc<X1DIR>(kp, jp, ip);
            const Real dx_j = coords.template Dxc<X2DIR>(kp, jp, ip);
            const Real dx_k = coords.template Dxc<X3DIR>(kp, jp, ip);
            auto rng_gen = rng_pool.get_state();
            const Real xl = coords.template Xc<parthenon::X1DIR>(ip) - 0.5 * dx_i;
            const Real xu = coords.template Xc<parthenon::X1DIR>(ip) + 0.5 * dx_i;
            const Real yl = coords.template Xc<parthenon::X2DIR>(jp) - 0.5 * dx_j;
            const Real yu = coords.template Xc<parthenon::X2DIR>(jp) + 0.5 * dx_j;
            const Real zl = coords.template Xc<parthenon::X3DIR>(kp) - 0.5 * dx_k;
            const Real zu = coords.template Xc<parthenon::X3DIR>(kp) + 0.5 * dx_k;

            Real ee_f = 0.;
            Real spower = 0.;
            Real inv_vx = (vx != 0.0) ? 1.0 / std::abs(vx) : 1e20;
            Real inv_vy = (vy != 0.0) ? 1.0 / std::abs(vy) : 1e20;
            Real inv_vz = (vz != 0.0) ? 1.0 / std::abs(vz) : 1e20;
            Real dt_remain = (t_start + dt) - t;
            if (dt_remain <= 0.0) {
              break;
            }

            int ip_c = std::max(ib.s, std::min(ip, ib.e));
            int jp_c = std::max(jb.s, std::min(jp, jb.e));
            int kp_c = std::max(kb.s, std::min(kp, kb.e));

            if (use_density_linear) {
              const Real rho = vmesh(b, fjh::density(), kp_c, jp_c, ip_c);
              const DensityLinearStoppingPower density_sp(
                  stopping_power_value, stopping_power_density_reference,
                  stopping_power_density_slope);
              spower = density_sp.Value(rho);
            } else if (use_temperature_linear) {
              const Real temperature = vmesh(b, fjh::temperature(), kp_c, jp_c, ip_c);
              const TemperatureLinearStoppingPower temperature_sp(
                  stopping_power_value, stopping_power_temperature_reference,
                  stopping_power_temperature_slope);
              spower = temperature_sp.Value(temperature);
            } else if (!use_lp) {
              spower = stopping_power_value;
            } else {
              Real rho = vmesh(b, fjh::density(), kp_c, jp_c, ip_c);
              Real na_local = rho / (ABAR * MPROTON);
              Real ne_local = Ze * na_local;
              Real Tbeta = vmesh(b, fjh::temperature(), kp_c, jp_c, ip_c);

              LiPetrassoStoppingPower LP_sp(Z, ne_local, na_local, Tbeta);
              spower = LP_sp.LP_Spower(ee);
            }

            Real vv_i = std::sqrt((2.0 * ee) / MALPHA);
            Real vv_f = std::sqrt((2.0 * EB) / MALPHA);
            Real dx_to_x = (vx > 0.0) ? (xu - x) * inv_vx * vv_i
                                      : ((vx < 0.0) ? (x - xl) * inv_vx * vv_i : 1e20);
            Real dx_to_y =
                multi_d ? ((vy > 0.0) ? (yu - y) * inv_vy * vv_i
                                      : ((vy < 0.0) ? (y - yl) * inv_vy * vv_i : 1e20))
                        : 1e20;
            Real dx_to_z =
                three_d ? ((vz > 0.0) ? (zu - z) * inv_vz * vv_i
                                      : ((vz < 0.0) ? (z - zl) * inv_vz * vv_i : 1e20))
                        : 1e20;
            Real dx_to_eb = (spower > 0.0) ? (ee - EB) / spower : 1e20;

            Real dx_to_dt = vv_i * dt_remain;
            if (spower > 0.0) {
              Real vv_end = vv_i - (spower / MALPHA) * dt_remain;
              const Real vv_background = std::sqrt((2.0 * EB) / MALPHA);
              vv_end = std::max(vv_end, vv_background);
              dx_to_dt = 0.5 * (vv_i + vv_end) * dt_remain;
            }

            Real ds = std::min({dx_to_x, dx_to_y, dx_to_z, dx_to_dt, dx_to_eb});

            PARTHENON_REQUIRE(!use_quad,
                              "Please don't use the quadratic model right now");
            ee_f = std::max(EB, ee - spower * ds);
            E_dep = (ee - ee_f) * ww;

            vv_f = std::sqrt(2.0 * ee_f / MALPHA);
            bool in_background = (ee_f <= EB + EPS);

            tran_step_args tra{
                // constants
                rng_gen, // rng
                t_start,
                dt, // timing
                ee,
                ee_f, // energy
                vv_i,
                vv_f,
                vx,
                vy,
                vz, // velocities
                ds,
                multi_d,
                three_d, // push, dimensionality
                xl,
                yl,
                zl,
                xu,
                yu,
                zu,     // coordinates
                spower, // stopping power value
                t,
                x,
                y,
                z,             // particle description
                in_background, // thermalization
            };

            ptcl_transport_step(tra);
            rng_pool.free_state(rng_gen);

            ppack_r(b, pa::energy(), n) = ee_f;
            const Real scale_velo = (vv_i > 0.0) ? vv_f / vv_i : 0.0;
            vx *= scale_velo;
            vy *= scale_velo;
            vz *= scale_velo;
            thermal = in_background;
            segment_distance = ds;
            segment_stopping = spower;
          }

          if constexpr (Plasma) {
            Kokkos::atomic_add(&vmesh(b, fj::electron_heat_step(), kp, jp, ip), ww * qe);
            Kokkos::atomic_add(&vmesh(b, fj::ion_heat_step(), kp, jp, ip), ww * qi);
            Kokkos::atomic_add(&vmesh(b, fj::electron_heat(), kp, jp, ip), ww * qe);
            Kokkos::atomic_add(&vmesh(b, fj::ion_heat(), kp, jp, ip), ww * qi);
            E_dep = ww * (qe + qi);
            ee = 0;
            for (int d = 0; d < 3; ++d)
              ee +=
                  0.5 * units::MALPHA * ppack_r(b, pa::v(d), n) * ppack_r(b, pa::v(d), n);
          }
          if constexpr (Compact)
            Kokkos::atomic_add(&heat(b, fj::deposited_energy_step(), kp, jp, ip), E_dep);
          else
            Kokkos::atomic_add(&heat(b, fj::energy_tally_step(), kp, jp, ip), E_dep);
          if constexpr (!Plasma && !Compact) {
            if (tally_mass_momentum) {
              for (int d = 0; d < 3; ++d)
                Kokkos::atomic_add(&vmesh(b, fj::momentum_delta_step(d), kp, jp, ip),
                                   ww * units::MALPHA *
                                       (old_velocity[d] - ppack_r(b, pa::v(d), n)));
            }
          }
          if constexpr (Diagnostics && !Plasma) {
            phi_local += ww * segment_distance / dv;
            e_density_accum += ww * 0.5 * (ee_i + ee) * (t - t_prev) / dv;
            residence_density_accum += ww * (t - t_prev) / dv;
            Kokkos::atomic_add(&diag(b, fj::spower_sum(), kp, jp, ip), segment_stopping);
            Kokkos::atomic_add(&diag(b, fj::spower_count(), kp, jp, ip), Real(1));
          }

          const bool escaped = !thermal && outside_global();
          if (!thermal && !escaped) {
            swarm_d.GetNeighborBlockIndex(n, x, y, z, on_block);
            if (on_block) swarm_d.Xtoijk(x, y, z, ip, jp, kp);
          }
          if constexpr (Diagnostics && !Plasma) {
            const bool cell_changed = ip != ip_p || jp != jp_p || kp != kp_p;
            if (cell_changed || !on_block || thermal || escaped || t >= end) {
              if (phi_local > 0) {
                Kokkos::atomic_add(&vmesh(b, fj::Phi(), kp_p, jp_p, ip_p), phi_local);
                Kokkos::atomic_add(&vmesh(b, fj::Phi_uq(), kp_p, jp_p, ip_p),
                                   phi_local * phi_local);
                Kokkos::atomic_add(&vmesh(b, fj::ave_energy(), kp_p, jp_p, ip_p),
                                   e_density_accum);
                Kokkos::atomic_add(&vmesh(b, fj::count_tally(), kp_p, jp_p, ip_p), ww);
                Kokkos::atomic_add(&vmesh(b, fj::residence_tally(), kp_p, jp_p, ip_p),
                                   residence_density_accum);
              }
              phi_local = e_density_accum = residence_density_accum = 0;
              ip_p = ip;
              jp_p = jp;
              kp_p = kp;
            }
          }
          if (thermal || escaped) {
            if constexpr (Plasma) {
              if (escaped) {
                Kokkos::atomic_add(&vmesh(b, fj::escaped_alphas(), kp, jp, ip), ww);
                Kokkos::atomic_add(&vmesh(b, fj::escaped_energy(), kp, jp, ip), ww * ee);
              }
            } else if (thermal) {
              if constexpr (Compact)
                Kokkos::atomic_add(&heat(b, fj::deposited_energy_step(), kp, jp, ip),
                                   ww * ee);
              else {
                Kokkos::atomic_add(&vmesh(b, fj::energy_delta_step(), kp, jp, ip),
                                   ww * ee);
                if (tally_mass_momentum) {
                  Kokkos::atomic_add(&vmesh(b, fj::mass_delta_step(), kp, jp, ip),
                                     ww * units::MALPHA);
                  for (int d = 0; d < 3; ++d)
                    Kokkos::atomic_add(&vmesh(b, fj::momentum_delta_step(d), kp, jp, ip),
                                       ww * units::MALPHA * ppack_r(b, pa::v(d), n));
                }
              }
            }
            swarm_d.MarkParticleForRemoval(n);
            break;
          }
        }
      });
  for (int b = 0; b < md->NumBlocks(); ++b)
    md->GetSwarmData(b)->Get(alphas_swarm_name)->RemoveMarkedParticles();
  if (time_transport) {
    Kokkos::fence();
    *cpt_pkg->MutableParam<Real>("transport_walltime_interval") +=
        std::chrono::duration<Real>(std::chrono::steady_clock::now() -
                                    transport_wall_start)
            .count();
  }
  return TaskStatus::complete;
}
} // namespace

TaskStatus TransportAlphas(MeshData<Real> *md, const Real t_start, const Real dt) {
  const auto &pkg = md->GetParentPointer()->packages.Get("charon");
  const bool plasma = pkg->Param<bool>("tn_plasma");
  if (pkg->Param<bool>("compact_tallies"))
    return plasma ? TransportAlphasImpl<false, true, true>(md, t_start, dt)
                  : TransportAlphasImpl<false, true, false>(md, t_start, dt);
  if (pkg->Param<bool>("cpt_diagnostics"))
    return plasma ? TransportAlphasImpl<true, false, true>(md, t_start, dt)
                  : TransportAlphasImpl<true, false, false>(md, t_start, dt);
  return plasma ? TransportAlphasImpl<false, false, true>(md, t_start, dt)
                : TransportAlphasImpl<false, false, false>(md, t_start, dt);
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus CheckCompletion
//! \brief Checks all particles on this mesh to see if they have reached the end
//!        of the timestep. If not, further iterations of transport are indicated.
TaskStatus CheckCompletion(MeshData<Real> *md, const Real t_end) {
  namespace pa = particle::alphas;
  namespace sp = swarm_position;

  // Create SwarmPacks
  static auto pdesc_r =
      MakeSwarmPackDescriptor<sp::x, sp::y, sp::z, pa::energy, pa::weight, pa::time>(
          alphas_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  const int &nparticles_per_pack = ppack_r.GetMaxFlatIndex();

  int num_unfinished = 0;
  parthenon::par_reduce(
      "CheckCompletion", 0, nparticles_per_pack,
      KOKKOS_LAMBDA(const int idx, int &num_unfinished) {
        auto [b, n] = ppack_r.GetBlockParticleIndices(idx);
        const auto &swarm_d = ppack_r.GetContext(b);
        if (swarm_d.IsActive(n)) {
          Real x = ppack_r(b, sp::x(), n);
          Real y = ppack_r(b, sp::y(), n);
          Real z = ppack_r(b, sp::z(), n);
          bool on_current_mesh_block;
          swarm_d.GetNeighborBlockIndex(n, x, y, z, on_current_mesh_block);

          if (ppack_r(b, pa::time(), n) < t_end && on_current_mesh_block) {
            num_unfinished++;
          }
        }
      },
      Kokkos::Sum<int>(num_unfinished));

  if (num_unfinished > 0) {
    return TaskStatus::iterate;
  } else {
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! template instantiations
TaskStatus TransportAlphas(MeshData<Real> *md, const Real t_start, const Real dt);

} // namespace charon
