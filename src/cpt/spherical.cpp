// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.
#include "cpt/spherical.hpp"
#include "cpt/spherical_geometry.hpp"

#include "charon.hpp"
#include "riot_utils/riot_utils.hpp"
#include "stopping_power.hpp"
#include "source_profiles.hpp"
#include "variables.hpp"
#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <parthenon_mpi.hpp>
#include <random>
#include <sstream>
#include <vector>

namespace CPT::spherical {
using namespace parthenon::driver::prelude;
namespace fj = field::charon;
namespace cb = cell_variables::cell_averaged::bulk;
namespace {
struct Particle {
  Vector x{}, v{};
  Real time = 0, weight = 0;
};
struct State {
  std::vector<Particle> particles;
  std::mt19937_64 rng;
  // Global round-robin source sequence, identical on every rank.
  unsigned long long source_sequence = 0;
  std::string checkpoint_directory;
};
// Background rows carry physical shell geometry and the existing plasma adapter.
enum Background { Inner, Outer, Volume, Ur, Ne, Te, Ti, Screening, Source, Birth, Ions };
enum Tally {
  Qe,
  Qi,
  Mass,
  Momentum,
  Energy,
  Thermal,
  Escape,
  EscapeEnergy,
  StopSum,
  StopCount,
  ActiveCount,
  ActiveWeight,
  ActiveEnergy,
  NTally
};
struct BackgroundData {
  int stride = 0;
  std::vector<Real> rows;       // sorted by actual inner face
  std::vector<int> local_shell; // local (partition, block, i) -> sorted shell
  int Size() const { return rows.size() / stride; }
  Real Get(int shell, int field) const { return rows[shell * stride + field]; }
};
std::shared_ptr<State> GetState(Mesh *pm) {
  return pm->packages.Get("charon")->Param<std::shared_ptr<State>>("spherical_state");
}
void Sum(std::vector<Real> &values) {
  PARTHENON_REQUIRE(values.size() <= INT_MAX, "Spherical CPT MPI tally too large");
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(),
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
}

BackgroundData Gather(Mesh *pm, bool material = true) {
  const bool plasma = pm->packages.Get("charon")->Param<bool>("tn_plasma");
  const int ns = plasma ? pm->packages.Get("charon")->Param<int>("plasma_num_species") : 0;
  BackgroundData bg;
  bg.stride = Ions + 2 * ns;
  const int stride = bg.stride;
  std::vector<Real> local;
  for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
    auto md = pm->mesh_data.GetOrAdd("base", p);
    auto geometry = riot::MakePack<cb::cpt_alpha_source>(md.get());
    const auto ib = md->GetBoundsI(IndexDomain::interior);
    const int j = md->GetBoundsJ(IndexDomain::interior).s;
    const int k = md->GetBoundsK(IndexDomain::interior).s;
    const int ni = ib.e - ib.s + 1, nb = geometry.GetNBlocks();
    if (nb == 0) continue;
    ParArray2D<Real> staging("Spherical CPT background", nb * ni, stride);
    parthenon::par_for(
        "GatherSphericalGeometry", 0, nb * ni - 1, KOKKOS_LAMBDA(int n) {
          const int b = n / ni, i = ib.s + n % ni;
          const auto &c = geometry.GetCoordinates(b);
          staging(n, Inner) = c.template Xf<X1DIR>(k, j, i);
          staging(n, Outer) = c.template Xf<X1DIR>(k, j, i + 1);
          staging(n, Volume) = c.CellVolume(k, j, i);
          for (int f = Ur; f < stride; ++f) staging(n, f) = 0;
        });
    if (plasma && material) {
      auto v = riot::MakePack<
          fj::plasma_velocity, fj::plasma_electron_density, fj::plasma_electron_temperature,
          fj::plasma_ion_temperature, fj::plasma_screening, fj::plasma_ion_density,
          fj::plasma_ion_charge, fj::source_alpha_energy, cb::cpt_alpha_source>(md.get());
      parthenon::par_for(
          "GatherSphericalPlasma", 0, nb * ni - 1, KOKKOS_LAMBDA(int n) {
            const int b = n / ni, i = ib.s + n % ni;
            staging(n, Ur) = v(b, fj::plasma_velocity(0), k, j, i);
            staging(n, Ne) = v(b, fj::plasma_electron_density(), k, j, i);
            staging(n, Te) = v(b, fj::plasma_electron_temperature(), k, j, i);
            staging(n, Ti) = v(b, fj::plasma_ion_temperature(), k, j, i);
            staging(n, Screening) = v(b, fj::plasma_screening(), k, j, i);
            staging(n, Source) = v(b, cb::cpt_alpha_source(), k, j, i);
            staging(n, Birth) = v(b, fj::source_alpha_energy(), k, j, i);
            for (int s = 0; s < ns; ++s) {
              staging(n, Ions + 2 * s) = v(b, fj::plasma_ion_density(s), k, j, i);
              staging(n, Ions + 2 * s + 1) = v(b, fj::plasma_ion_charge(s), k, j, i);
            }
          });
    }
    auto h = staging.GetHostMirrorAndCopy();
    for (int n = 0; n < nb * ni; ++n)
      for (int f = 0; f < stride; ++f)
        local.push_back(h(n, f));
  }
  PARTHENON_REQUIRE(local.size() <= INT_MAX, "Spherical background too large");
  int count = local.size();
  std::vector<int> counts(Globals::nranks), offsets(Globals::nranks, 0);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD));
#else
  counts[0] = count;
#endif
  long long total = 0;
  for (int r = 0; r < Globals::nranks; ++r) {
    offsets[r] = total;
    total += counts[r];
    PARTHENON_REQUIRE(total <= INT_MAX, "Spherical background exceeds MPI count limit");
  }
  std::vector<Real> gathered(total);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allgatherv(local.data(), count, MPI_PARTHENON_REAL,
                                     gathered.data(), counts.data(), offsets.data(),
                                     MPI_PARTHENON_REAL, MPI_COMM_WORLD));
#else
  gathered = local;
#endif
  const int n = total / stride;
  PARTHENON_REQUIRE(n > 0, "Empty radial mesh");
  std::vector<int> order(n), inverse(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return gathered[a * stride + Inner] < gathered[b * stride + Inner];
  });
  bg.rows.resize(total);
  for (int s = 0; s < n; ++s) {
    inverse[order[s]] = s;
    std::copy_n(gathered.begin() + order[s] * stride, stride,
                bg.rows.begin() + s * stride);
    const Real lo = bg.Get(s, Inner), hi = bg.Get(s, Outer);
    const Real full_volume = 4 * M_PI / 3 * (hi * hi * hi - lo * lo * lo);
    PARTHENON_REQUIRE(lo >= 0 && hi > lo &&
                          std::abs(bg.Get(s, Volume) / full_volume - 1) < 1.e-9,
                      "Spherical CPT requires full-sphere cell volumes");
    PARTHENON_REQUIRE(
        s == 0 ? lo == 0 : std::abs(lo - bg.Get(s - 1, Outer)) < 1.e-10 * hi,
        "Radial mesh must tile [0,R] without gaps or overlapping coarse cells");
  }
  const int first = offsets[Globals::my_rank] / stride;
  for (int l = 0; l < count / stride; ++l)
    bg.local_shell.push_back(inverse[first + l]);
  return bg;
}

int Locate(const BackgroundData &bg, Real radius) {
  int lo = 0, hi = bg.Size();
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (radius < bg.Get(mid, Outer))
      hi = mid;
    else
      lo = mid + 1;
  }
  return lo;
}
Vector FluidVelocity(const BackgroundData &bg, int s, const Vector &x) {
  Vector u{};
  const Real radius = Radius(x);
  // In the innermost cell extend the cell-average speed linearly to zero
  // inside its volume centroid, giving a regular vector field at the origin.
  const Real denominator = s == 0 ? std::max(radius, 0.75 * bg.Get(s, Outer)) : radius;
  if (denominator > 0)
    for (int d = 0; d < 3; ++d)
      u[d] = bg.Get(s, Ur) * x[d] / denominator;
  return u;
}
Real Uniform(State &state) { return std::generate_canonical<Real, 53>(state.rng); }
Vector Isotropic(State &state) {
  const Real mu = 2 * Uniform(state) - 1, phi = 2 * M_PI * Uniform(state);
  const Real transverse = std::sqrt(std::max(Real(0), 1 - mu * mu));
  return {transverse * std::cos(phi), transverse * std::sin(phi), mu};
}

void SourceVerification(Mesh *pm, BackgroundData &bg, Real start, Real dt) {
  auto pkg = pm->packages.Get("charon");
  auto state = GetState(pm);
  const auto strategy = pkg->Param<charon::SourceStrategy>("source_strategy");
  const int initial = pkg->Param<int>("num_particles");
  const int amplitude = pkg->Param<int>("num_particles_per_pulse");
  const Real final_time = pkg->Param<Real>("tlim");
  int count = start == 0 ? initial : amplitude;
  if (pkg->Param<std::string>("variable_field") == "sine_test") {
    count = charon::SineTestParticleCount(start, final_time, initial, amplitude);
  } else if (pkg->Param<int>("pulses") > 0) {
    const int pulses = pkg->Param<int>("pulses");
    const Real interval = final_time / pulses;
    const Real eps = 16 * std::numeric_limits<Real>::epsilon() *
                     std::max(std::abs(start) + std::abs(dt), interval);
    const int first = std::max(0, static_cast<int>(std::ceil((start - eps) / interval)));
    const int last = std::min(pulses - 1,
                             static_cast<int>(std::floor((start + dt - eps) / interval)));
    count = 0;
    for (int pulse = first; pulse <= last; ++pulse)
      count += pulse == 0 ? initial : amplitude;
  }
  const Real weight = pkg->Param<Real>("alpha_weight");
  const Real speed = std::sqrt(2 * charon::units::E0 / charon::units::MALPHA);
  const auto first = FirstSourceForRank(state->source_sequence, Globals::my_rank,
                                       Globals::nranks);
  std::vector<Real> shell_counts(bg.Size(), 0);
  for (auto n = first; n < static_cast<unsigned long long>(count); n += Globals::nranks) {
    Particle a;
    if (strategy == charon::SourceStrategy::isotropic) {
      a.x = Isotropic(*state);
      const Real r = BirthRadius(0, bg.Get(bg.Size() - 1, Outer), Uniform(*state));
      for (auto &x : a.x) x *= r;
    } // Point particles are born exactly at the origin.
    const auto direction = Isotropic(*state);
    for (int d = 0; d < 3; ++d) a.v[d] = speed * direction[d];
    a.time = start; // Same timestep-start births as non-plasma Cartesian sourcing.
    a.weight = weight;
    state->particles.push_back(a);
    shell_counts[Locate(bg, Radius(a.x))] += 1;
  }
  state->source_sequence += count;
  Sum(shell_counts);
  for (int shell = 0; shell < bg.Size(); ++shell)
    bg.rows[shell * bg.stride + Source] = shell_counts[shell];
  pkg->UpdateParam<Real>("cumulative_alphas_sourced",
                         pkg->Param<Real>("cumulative_alphas_sourced") + count * weight);
}

// Exact constant-drag characteristics in a stationary medium. Cell intersections
// partition the deposited energy; reflection preserves the homogeneous 0D inventory.
void TransportVerification(Mesh *pm, const BackgroundData &bg, Real end,
                           std::vector<Real> &tallies) {
  auto pkg = pm->packages.Get("charon");
  auto state = GetState(pm);
  const Real stopping = pkg->Param<Real>("stopping_power_value");
  const Real mass = charon::units::MALPHA;
  const bool reflect = pkg->Param<bool>("verification_reflect_outer");
  std::size_t survivor = 0;
  for (auto a : state->particles) {
    bool removed = false;
    int iterations = 0;
    while (a.time < end) {
      PARTHENON_REQUIRE(++iterations <= pkg->Param<int>("max_transport_iterations"),
                        "Spherical verification exceeded max_transport_iterations");
      const int shell = Locate(bg, Radius(a.x));
      PARTHENON_REQUIRE(shell < bg.Size(), "Invalid spherical verification position");
      auto *t = tallies.data() + shell * NTally;
      const Real speed = Radius(a.v);
      if (speed == 0) { removed = true; break; }
      Vector direction = a.v;
      for (auto &v : direction) v /= speed;
      const Real wall = ShellCrossing(a.x, direction, bg.Get(shell, Inner), bg.Get(shell, Outer));
      const auto step = ConstantDragSegment(speed, mass, stopping, end - a.time, wall);
      for (int d = 0; d < 3; ++d) {
        a.x[d] += direction[d] * step.distance;
        a.v[d] = direction[d] * step.speed;
      }
      a.time += step.dt;
      t[Qe] += a.weight * step.loss;
      t[StopSum] += stopping;
      t[StopCount] += 1;
      if (step.speed == 0) {
        t[Thermal] += a.weight;
        removed = true;
        break;
      }
      if (step.hit_wall) {
        const Real radius = Radius(a.x);
        const Real epsilon = 1.e-10 * (bg.Get(shell, Outer) - bg.Get(shell, Inner));
        const bool outward = Dot(a.x, a.v) > 0;
        Real target = radius + (outward ? epsilon : -epsilon);
        if (outward && shell == bg.Size() - 1) {
          if (!reflect) {
            t[Escape] += a.weight;
            t[EscapeEnergy] += a.weight * 0.5 * mass * step.speed * step.speed;
            removed = true;
            break;
          }
          const Real projection = Dot(a.x, a.v) / (radius * radius);
          for (int d = 0; d < 3; ++d) a.v[d] -= 2 * projection * a.x[d];
          target = radius - epsilon;
        }
        for (auto &x : a.x) x *= target / radius;
      } else {
        a.time = end;
      }
    }
    if (!removed) { a.time = end; state->particles[survivor++] = a; }
  }
  state->particles.resize(survivor);
}

void SourceParticles(Mesh *pm, BackgroundData &bg, Real start, Real dt) {
  if (!pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
    SourceVerification(pm, bg, start, dt);
    return;
  }
  auto state = GetState(pm);
  const Real weight = pm->packages.Get("charon")->Param<Real>("alpha_weight");
  const unsigned long long nranks = Globals::nranks, rank = Globals::my_rank;
  unsigned long long stage_histories = 0;
  for (int s = 0; s < bg.Size(); ++s) {
    const Real source = bg.Get(s, Source);
    PARTHENON_REQUIRE(std::isfinite(source) && source >= 0 && source <= INT_MAX &&
                          source == std::floor(source),
                      "Invalid radial source count");
    const unsigned long long count = source;
    stage_histories += count;
    const auto first = FirstSourceForRank(state->source_sequence, rank, nranks);
    for (auto n = first; n < count; n += nranks) {
      Particle a;
      a.x = Isotropic(*state);
      const Real radius =
          BirthRadius(bg.Get(s, Inner), bg.Get(s, Outer), Uniform(*state));
      for (auto &x : a.x)
        x *= radius;
      const Vector u = FluidVelocity(bg, s, a.x), direction = Isotropic(*state);
      const Real speed = std::sqrt(2 * bg.Get(s, Birth) / charon::units::MALPHA);
      for (int d = 0; d < 3; ++d)
        a.v[d] = u[d] + speed * direction[d];
      a.time = start + Uniform(*state) * dt;
      a.weight = weight;
      state->particles.push_back(a);
    }
    state->source_sequence += count;
  }
  Real sampled = 0;
  for (int s = 0; s < bg.Size(); ++s)
    sampled += bg.Get(s, Source) * weight;
  auto pkg = pm->packages.Get("charon");
  pkg->UpdateParam<Real>("cumulative_alphas_sourced",
                         pkg->Param<Real>("cumulative_alphas_sourced") + sampled);
  // Background source counts are already global and replicated. Do not sum
  // them over ranks again. Match the Cartesian source log before transport.
  if (Globals::my_rank == 0 && stage_histories > 0) {
    std::cout << "[Charon Sourcing] | Geometry: spherical"
              << " | Time: " << start << " | Sourced this stage: " << stage_histories
              << " | Cumulative weighted alphas: "
              << pkg->Param<Real>("cumulative_alphas_sourced") << std::endl;
  }
}

void Transport(Mesh *pm, const BackgroundData &bg, Real end, std::vector<Real> &tallies) {
  if (!pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
    TransportVerification(pm, bg, end, tallies);
    return;
  }
  auto pkg = pm->packages.Get("charon");
  auto state = GetState(pm);
  const bool lp = pkg->Param<std::string>("stopping_power_model") == "LiPetrasso";
  const Real constant = pkg->Param<Real>("stopping_power_value");
  const Real ion_fraction = pkg->Param<Real>("constant_ion_fraction");
  const Real cutoff = pkg->Param<Real>("plasma_thermalization_energy");
  const Real tolerance = pkg->Param<Real>("plasma_velocity_tolerance");
  const Real fraction = pkg->Param<Real>("plasma_fractional_step");
  const int limit = pkg->Param<int>("plasma_max_substeps");
  auto species_mass = pkg->Param<ParArray1D<Real>>("plasma_species_mass");
  const auto mass = species_mass.GetHostMirrorAndCopy();
  const int ns = pkg->Param<int>("plasma_num_species");
  std::size_t survivor = 0;
  // CPU rank-local transport. Stopping and drag integration are shared with 3D.
  for (auto a : state->particles) {
    int steps = 0;
    bool removed = false;
    charon::TransportClock clock(a.time, end);
    while (clock.remaining > 0) {
      PARTHENON_REQUIRE(++steps <= limit, "Spherical CPT substep limit exceeded");
      const int s = Locate(bg, Radius(a.x));
      PARTHENON_REQUIRE(s < bg.Size(), "Census particle outside spherical domain");
      Real *t = tallies.data() + s * NTally;
      const Vector u = FluidVelocity(bg, s, a.x), old_v = a.v, old_x = a.x;
      const Real old_energy = 0.5 * charon::units::MALPHA * Dot(a.v, a.v);
      Vector direction{};
      for (int d = 0; d < 3; ++d)
        direction[d] = a.v[d] - u[d];
      const Real speed = Radius(direction),
                 energy = 0.5 * charon::units::MALPHA * speed * speed;
      PARTHENON_REQUIRE(std::isfinite(energy), "Invalid relative alpha energy");
      const bool interacting = bg.Get(s, Screening) > 0 || (!lp && constant > 0);
      bool thermal = interacting && energy <= cutoff * (1 + 1.e-8);
      Real qe = 0, qi = thermal ? energy : 0;
      if (!thermal) {
        const charon::LPContext electron_context{Radius(a.x), clock.Time(),
                                                 Globals::my_rank, s, -1};
        if (speed > 0)
          for (auto &d : direction)
            d /= speed;
        const auto stopping = [&](Real e, Real &se, Real &si) {
          if (!lp) {
            se = constant * (1 - ion_fraction);
            si = constant * ion_fraction;
            return;
          }
          se = charon::PlasmaStopping::Channel(e, bg.Get(s, Ne), charon::units::MBETA, 1,
                                               bg.Get(s, Te), bg.Get(s, Screening),
                                               electron_context);
          si = 0;
          for (int j = 0; j < ns; ++j) {
            auto ion_context = electron_context;
            ion_context.species = j;
            si += charon::PlasmaStopping::Channel(
                e, bg.Get(s, Ions + 2 * j), mass(j), bg.Get(s, Ions + 2 * j + 1),
                bg.Get(s, Ti), bg.Get(s, Screening), ion_context);
          }
        };
        Real se, si;
        stopping(energy, se, si);
        Real h = clock.remaining;
        const Real total = se + si;
        if (total > 0) {
          h = std::min(h, fraction * charon::units::MALPHA * speed / total);
          h = std::min(h, (energy - cutoff) / (total * speed));
        }
        // Resolve rotation of the radial fluid frame and drag curvature. A
        // segment is never allowed to traverse an entire shell unchecked.
        const Real width = bg.Get(s, Outer) - bg.Get(s, Inner);
        const Real lab_bound = speed + Radius(u);
        if (lab_bound > 0) {
          h = std::min(h, 0.25 * width / lab_bound);
          if (Radius(u) > 0)
            h = std::min(h,
                         std::sqrt(tolerance) * std::max(Radius(a.x), width) / lab_bound);
        }
        charon::DragStep step{};
        Vector displacement{};
        bool accepted = false;
        for (int retry = 0; retry < 64; ++retry) {
          step = speed > 0 ? charon::TakeDragStep(energy, interacting ? cutoff : Real(0),
                                                  h, tolerance, stopping)
                           : charon::DragStep{h, 0, 0, 0};
          h = step.dt;
          for (int d = 0; d < 3; ++d)
            displacement[d] = (u[d] + direction[d] * 0.5 * (speed + step.speed)) * h;
          const Real crossing =
              ShellCrossing(a.x, displacement, bg.Get(s, Inner), bg.Get(s, Outer));
          if (crossing >= 1 - 1.e-11) {
            accepted = true;
            break;
          }
          h *= crossing;
        }
        PARTHENON_REQUIRE(accepted && std::isfinite(h) && h > 0,
                          "Spherical CPT face integration failed");
        clock.Advance(h);
        const Real final_relative = 0.5 * charon::units::MALPHA * step.speed * step.speed;
        const Real loss = std::max(Real(0), energy - final_relative);
        const Real mid_total = step.electron_stopping + step.ion_stopping;
        qe = mid_total > 0 ? loss * step.electron_stopping / mid_total : 0;
        qi = loss - qe;
        for (int d = 0; d < 3; ++d) {
          a.x[d] += displacement[d];
          a.v[d] = u[d] + direction[d] * step.speed;
        }
        thermal = interacting && final_relative <= cutoff * (1 + 1.e-8);
        if (thermal) qi += final_relative;
        t[StopSum] += mid_total;
        t[StopCount] += 1;
      }
      const Real final_energy = 0.5 * charon::units::MALPHA * Dot(a.v, a.v);
      const auto transfer = charon::TransferToFluid(old_v.data(), a.v.data(), old_energy,
                                                    final_energy, a.weight, thermal);
      // The spherical momentum equation evolves radial momentum. Project at
      // the segment midpoint; transverse momentum averages away over the sphere.
      Vector midpoint{};
      for (int d = 0; d < 3; ++d)
        midpoint[d] = 0.5 * (a.x[d] + old_x[d]);
      t[Qe] += a.weight * qe;
      t[Qi] += a.weight * qi;
      t[Mass] += transfer.mass;
      t[Energy] += transfer.energy;
      const Real radius = Radius(a.x);
      const Real midpoint_radius = Radius(midpoint);
      if (midpoint_radius > 0)
        for (int d = 0; d < 3; ++d)
          t[Momentum] += transfer.momentum[d] * midpoint[d] / midpoint_radius;
      if (thermal) {
        t[Thermal] += a.weight;
        removed = true;
        break;
      }
      // Nudge only roundoff at a reached face, after tallying to the traversed
      // shell. Passage through r=0 requires no reflection or special removal.
      const Real radial = Dot(a.x, a.v);
      const Real epsilon = 1.e-10 * (bg.Get(s, Outer) - bg.Get(s, Inner));
      Real target = radius;
      if (radial > 0 && bg.Get(s, Outer) - radius <= epsilon)
        target = bg.Get(s, Outer) + epsilon;
      if (radial < 0 && bg.Get(s, Inner) > 0 && radius - bg.Get(s, Inner) <= epsilon)
        target = bg.Get(s, Inner) - epsilon;
      if (radius > 0)
        for (auto &x : a.x)
          x *= target / radius;
      if (target >= bg.Get(bg.Size() - 1, Outer)) {
        t[Escape] += a.weight;
        t[EscapeEnergy] += a.weight * final_energy;
        removed = true;
        break;
      }
    }
    if (!removed) {
      a.time = end;
      state->particles[survivor++] = a;
    }
  }
  state->particles.resize(survivor);
}

void ActiveTallies(Mesh *pm, const BackgroundData &bg, std::vector<Real> &tallies) {
  for (const auto &a : GetState(pm)->particles) {
    const int s = Locate(bg, Radius(a.x));
    PARTHENON_REQUIRE(s < bg.Size(), "Invalid spherical census position");
    tallies[s * NTally + ActiveCount] += 1;
    tallies[s * NTally + ActiveWeight] += a.weight;
    tallies[s * NTally + ActiveEnergy] +=
        a.weight * 0.5 * charon::units::MALPHA * Dot(a.v, a.v);
  }
}

template <bool Diagnostics, bool Compact>
void Scatter(Mesh *pm, const BackgroundData &bg, const std::vector<Real> &tallies,
             bool active_only = false) {
  std::size_t local = 0;
  for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
    auto md = pm->mesh_data.GetOrAdd("base", p);
    auto geometry = riot::MakePack<cb::cpt_alpha_source>(md.get());
    const auto ib = md->GetBoundsI(IndexDomain::interior);
    const int j = md->GetBoundsJ(IndexDomain::interior).s;
    const int k = md->GetBoundsK(IndexDomain::interior).s;
    const int ni = ib.e - ib.s + 1, nb = geometry.GetNBlocks();
    if (nb == 0) continue;
    ParArray2D<Real> staging("Spherical CPT reduced tallies", nb * ni, NTally + 1);
    auto h = staging.GetHostMirror();
    for (int n = 0; n < nb * ni; ++n) {
      const int s = bg.local_shell.at(local++);
      for (int f = 0; f < NTally; ++f)
        h(n, f) = tallies[s * NTally + f];
      h(n, NTally) = bg.Get(s, Source);
    }
    staging.DeepCopy(h);
    if (!active_only && pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
      auto v =
          riot::MakePack<fj::electron_heat_step, fj::ion_heat_step, fj::deposited_mass_step,
                       fj::deposited_momentum_step, fj::deposited_lab_energy_step,
                       fj::electron_heat, fj::ion_heat, fj::thermalized_alphas,
                       fj::escaped_alphas, fj::escaped_energy>(md.get());
      parthenon::par_for(
          "ScatterSphericalCPT", 0, nb * ni - 1, KOKKOS_LAMBDA(int n) {
            const int b = n / ni, i = ib.s + n % ni;
            v(b, fj::electron_heat_step(), k, j, i) = staging(n, Qe);
            v(b, fj::ion_heat_step(), k, j, i) = staging(n, Qi);
            v(b, fj::deposited_mass_step(), k, j, i) = staging(n, Mass);
            v(b, fj::deposited_lab_energy_step(), k, j, i) = staging(n, Energy);
            for (int d = 0; d < 3; ++d)
              v(b, fj::deposited_momentum_step(d), k, j, i) =
                  d == 0 ? staging(n, Momentum) : 0;
            v(b, fj::electron_heat(), k, j, i) += staging(n, Qe);
            v(b, fj::ion_heat(), k, j, i) += staging(n, Qi);
            v(b, fj::thermalized_alphas(), k, j, i) += staging(n, Thermal);
            v(b, fj::escaped_alphas(), k, j, i) += staging(n, Escape);
            v(b, fj::escaped_energy(), k, j, i) += staging(n, EscapeEnergy);
          });
    }
    if constexpr (Diagnostics) {
      auto diag = riot::MakePack<fj::active_num_per_cell, fj::active_ew_per_cell,
                                 fj::active_ke_tally, fj::source_num_per_cell,
                                 fj::source_ew_per_cell, fj::spower_sum, fj::spower_count,
                                 fj::spower_tally>(md.get());
      const Real weight = pm->packages.Get("charon")->Param<Real>("alpha_weight");
      parthenon::par_for(
          "SphericalCPTDiagnostics", 0, nb * ni - 1, KOKKOS_LAMBDA(int n) {
            const int b = n / ni, i = ib.s + n % ni;
            diag(b, fj::active_num_per_cell(), k, j, i) = staging(n, ActiveCount);
            diag(b, fj::active_ew_per_cell(), k, j, i) = staging(n, ActiveWeight);
            diag(b, fj::active_ke_tally(), k, j, i) = staging(n, ActiveEnergy);
            if (!active_only) {
              diag(b, fj::source_num_per_cell(), k, j, i) = staging(n, NTally);
              diag(b, fj::source_ew_per_cell(), k, j, i) = staging(n, NTally) * weight;
              diag(b, fj::spower_sum(), k, j, i) += staging(n, StopSum);
              diag(b, fj::spower_count(), k, j, i) += staging(n, StopCount);
            }
            const Real count = diag(b, fj::spower_count(), k, j, i);
            diag(b, fj::spower_tally(), k, j, i) =
                count > 0 ? diag(b, fj::spower_sum(), k, j, i) / count : 0;
          });
    }
    if constexpr (Compact) {
      if (!active_only && !pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
        auto heat = riot::MakePack<fj::deposited_energy_step>(md.get());
        parthenon::par_for("SphericalVerificationHeat", 0, nb * ni - 1,
            KOKKOS_LAMBDA(int n) {
              heat(n / ni, fj::deposited_energy_step(), k, j, ib.s + n % ni) =
                  staging(n, Qe) + staging(n, Qi);
            });
      }
    }
    if constexpr (!Compact) {
      if (!active_only) {
        auto heat = riot::MakePack<fj::energy_tally, fj::energy_tally_step,
                                   fj::energy_delta_step>(md.get());
        parthenon::par_for(
            "SphericalCPTHeat", 0, nb * ni - 1, KOKKOS_LAMBDA(int n) {
              const int b = n / ni, i = ib.s + n % ni;
              const Real energy = staging(n, Qe) + staging(n, Qi);
              heat(b, fj::energy_tally_step(), k, j, i) = energy;
              heat(b, fj::energy_delta_step(), k, j, i) = 0;
              heat(b, fj::energy_tally(), k, j, i) += energy;
            });
      }
    }
  }
  Kokkos::fence();
}
void ScatterSelected(Mesh *pm, const BackgroundData &bg, std::vector<Real> &t,
                     bool active_only = false) {
  Sum(t);
  auto pkg = pm->packages.Get("charon");
  if (pkg->Param<bool>("cpt_diagnostics"))
    Scatter<true, false>(pm, bg, t, active_only);
  else if (pkg->Param<bool>("compact_tallies"))
    Scatter<false, true>(pm, bg, t, active_only);
  else
    Scatter<false, false>(pm, bg, t, active_only);
}

std::filesystem::path CheckpointPath(const State &state, int cycle) {
  std::ostringstream name;
  name << "cpt-spherical." << std::setw(10) << std::setfill('0') << cycle << ".rank"
       << std::setw(6) << Globals::my_rank << ".txt";
  return std::filesystem::path(state.checkpoint_directory) / name.str();
}
void Save(Mesh *pm, const SimTime &tm) {
  auto state = GetState(pm);
  const auto target = CheckpointPath(*state, tm.ncycle);
  const auto temporary = target.string() + ".tmp";
  std::ofstream out(temporary);
  out << std::setprecision(std::numeric_limits<Real>::max_digits10);
  out << "RIOT_SPHERICAL_CPT_V1 " << Globals::nranks << ' ' << Globals::my_rank << ' '
      << tm.ncycle << ' ' << tm.time << ' ' << state->source_sequence << ' '
      << state->particles.size() << '\n'
      << state->rng << '\n';
  for (const auto &a : state->particles) {
    for (auto x : a.x)
      out << x << ' ';
    for (auto v : a.v)
      out << v << ' ';
    out << a.time << ' ' << a.weight << '\n';
  }
  out.flush();
  PARTHENON_REQUIRE(out.good(), "Could not write spherical CPT checkpoint");
  out.close();
  std::filesystem::rename(temporary, target);
  // All companions must exist before the hydro checkpoint can be written.
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
}
} // namespace

void Initialize(ParameterInput *pin, StateDescriptor *pkg) {
  PARTHENON_REQUIRE(pin->GetInteger("parthenon/mesh", "nx2") == 1 &&
                        pin->GetInteger("parthenon/mesh", "nx3") == 1 &&
                        pin->GetString("parthenon/mesh", "refinement") == "none" &&
                        pin->GetReal("parthenon/mesh", "x1min") == 0,
                    "Spherical CPT requires a uniform 1D radial mesh starting at zero");
  PARTHENON_REQUIRE(
      !pkg->Param<bool>("weak_scaling_diagnostics") &&
          pkg->Param<bool>("transport_alphas"),
      "Spherical CPT requires transport_alphas=true and weak_scaling_diagnostics=false");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||                         \
    defined(KOKKOS_ENABLE_SYCL)
  PARTHENON_FAIL("Spherical CPT currently supports CPU builds only");
#endif
  if (!pkg->Param<bool>("tn_plasma")) {
    const auto strategy = pkg->Param<charon::SourceStrategy>("source_strategy");
    PARTHENON_REQUIRE(pkg->Param<std::string>("stopping_power_model") == "constant" &&
                          pin->GetOrAddBoolean("physics", "fixed_fluid", false) &&
                          !pkg->Param<bool>("do_population_control") &&
                          (strategy == charon::SourceStrategy::point ||
                           strategy == charon::SourceStrategy::isotropic),
                      "Spherical verification requires constant stopping, fixed_fluid, "
                      "no population control, and point or isotropic sourcing");
    PARTHENON_REQUIRE(pkg->Param<int>("num_particles") >= 0 &&
                          pkg->Param<int>("num_particles_per_pulse") >= 0 &&
                          std::isfinite(pkg->Param<Real>("tlim")) &&
                          pkg->Param<Real>("tlim") > 0,
                      "Spherical verification requires nonnegative counts and finite tlim > 0");
    PARTHENON_REQUIRE(std::isfinite(pkg->Param<Real>("stopping_power_value")) &&
                          pkg->Param<Real>("stopping_power_value") >= 0,
                      "Spherical verification requires nonnegative finite stopping power");
    const auto outer = pin->GetString("parthenon/mesh", "ox1_bc");
    PARTHENON_REQUIRE(outer == "reflecting" || outer == "outflow",
                      "Spherical verification supports reflecting or outflow outer boundaries");
    pkg->AddParam("verification_reflect_outer", outer == "reflecting");
  }
  auto state = std::make_shared<State>();
  state->rng.seed(pkg->Param<int>("seed"));
  state->checkpoint_directory =
      pin->GetOrAddString("charon", "spherical_checkpoint_directory",
                          std::filesystem::current_path().string());
  pkg->AddParam("spherical_state", state);
  const auto before = pkg->UserWorkBeforeLoopMesh;
  pkg->UserWorkBeforeLoopMesh = [before](Mesh *pm, ParameterInput *input, SimTime &tm) {
    before(pm, input, tm);
    auto state = GetState(pm);
    if (tm.ncycle != 0 || tm.time != 0) {
      std::ifstream in(CheckpointPath(*state, tm.ncycle));
      std::string magic;
      int ranks = -1, rank = -1, cycle = -1;
      Real time = -1;
      std::size_t count = 0;
      in >> magic >> ranks >> rank >> cycle >> time >> state->source_sequence >> count;
      PARTHENON_REQUIRE(in.good() && magic == "RIOT_SPHERICAL_CPT_V1" &&
                            ranks == Globals::nranks && rank == Globals::my_rank &&
                            cycle == tm.ncycle && time == tm.time,
                        "Missing/incompatible spherical CPT companion checkpoint; "
                        "restore its directory and use the original MPI rank count");
      in >> state->rng;
      state->particles.resize(count);
      for (auto &a : state->particles) {
        for (auto &x : a.x)
          in >> x;
        for (auto &v : a.v)
          in >> v;
        in >> a.time >> a.weight;
        PARTHENON_REQUIRE(in.good() && a.time == tm.time && a.weight > 0 &&
                              std::isfinite(Radius(a.x)) && std::isfinite(Radius(a.v)),
                          "Invalid spherical CPT companion particle state");
      }
      PARTHENON_REQUIRE(in.good(), "Truncated spherical CPT RNG/particle checkpoint");
    }
    // Load from the saved directory; future checkpoints belong to this run.
    state->checkpoint_directory = std::filesystem::current_path().string();
    input->SetString("charon", "spherical_checkpoint_directory",
                     state->checkpoint_directory);
    // Native work fields are not checkpointed. Initialize them and refresh
    // census diagnostics without reading an uninitialized plasma background.
    auto bg = Gather(pm, false);
    std::vector<Real> tallies(bg.Size() * NTally, 0);
    ActiveTallies(pm, bg, tallies);
    ScatterSelected(pm, bg, tallies);
  };
  pkg->UserWorkBeforeOutputMesh = [](Mesh *pm, ParameterInput *, SimTime &tm) {
    if (pm->packages.Get("charon")->Param<bool>("cpt_diagnostics")) {
      auto bg = Gather(pm, false);
      std::vector<Real> tallies(bg.Size() * NTally, 0);
      ActiveTallies(pm, bg, tallies);
      ScatterSelected(pm, bg, tallies, true);
    }
    Save(pm, tm);
  };
}

void Step(Mesh *pm, Real start, Real dt) {
  auto bg = Gather(pm);
  SourceParticles(pm, bg, start, dt);
  std::vector<Real> tallies(bg.Size() * NTally, 0);
  Transport(pm, bg, start + dt, tallies);
  ActiveTallies(pm, bg, tallies);
  ScatterSelected(pm, bg, tallies);
}
} // namespace CPT::spherical
