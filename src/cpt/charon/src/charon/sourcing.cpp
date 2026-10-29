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

// C++ includes
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

// Parthenon includes
#include <parthenon_mpi.hpp>
#include <utils/robust.hpp>

// Charon includes
#include "charon.hpp"
#include "charon_utils.hpp"
#include "transport_utils.hpp"
#include "charon_variables.hpp"
#include "source_profiles.hpp"
#include "units.hpp"

namespace charon {
using namespace units;

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus SourceAlphas
//! \brief Create alphas, either during initialization or during a timestep.
//! TODO(BRR) modify interface so we don't need t_start, dt for initialization
template <typename T, bool Diagnostics, bool Plasma = false>
TaskStatus SourceAlphasImpl(T *md, const Real t_start, const Real dt) {
  namespace fj = field::charon;
  namespace fjh = field::charon::host;
  namespace pa = particle::alphas; // Alpha particle name

  auto pm = md->GetParentPointer();
  if constexpr (Plasma) {
    if (dt <= 0.0) return TaskStatus::complete;
  }
  auto &resolved_pkgs = pm->resolved_packages;
//  auto &cpt_pkg = pm->packages.Get("charon");
  auto &cpt_pkg = md->GetParentPointer()->packages.Get("charon");
  const auto &strategy = cpt_pkg->template Param<SourceStrategy>("source_strategy");

  std::string alpha_file = cpt_pkg->template Param<std::string>("alpha_source_file");
  std::string alpha_mode = cpt_pkg->template Param<std::string>("alpha_source_mode");
  bool Qmcblock = false;
  if (alpha_file != "none" && pm->packages.AllPackages().count("mcblock")) {
    Qmcblock = true;
  }

  Real vv = CharonNull<Real>();
  Real vx_out = CharonNull<Real>();
  Real vy_out = CharonNull<Real>();
  Real vz_out = CharonNull<Real>();

  // Extract params
  auto &rng_pool = cpt_pkg->template Param<RngPool>("rng_pool");
  const int &num_particles = cpt_pkg->template Param<int>("num_particles");
  const int &num_particles_per_pulse = cpt_pkg->template Param<int>("num_particles_per_pulse");
  const bool &verbose = cpt_pkg->template Param<bool>("verbose");
  const Real &dnpc_min = cpt_pkg->template Param<Real>("dnpc_min"); // min number of MC particles to add per cell
  const Real a_ww = cpt_pkg->template Param<Real>("alpha_weight");
  const Real alpha_source_physical_cap =
      cpt_pkg->template Param<Real>("alpha_source_physical_cap");
  const Real fuel_density_threshold =
      cpt_pkg->template Param<Real>("fuel_density_threshold");

  auto desc = [&]() {
    if constexpr (Plasma && Diagnostics) {
      return MakePackDescriptor<fj::source_num_per_cell, fj::source_ew_per_cell,
                                fj::delta_num_per_cell, fjh::density,
                                fjh::alpha_source, fj::plasma_velocity>(resolved_pkgs.get());
    } else if constexpr (Plasma) {
      return MakePackDescriptor<fjh::density, fjh::alpha_source,
                                fj::plasma_velocity>(resolved_pkgs.get());
    } else if constexpr (Diagnostics) {
      return MakePackDescriptor<fj::source_num_per_cell, fj::source_ew_per_cell,
                                fj::delta_num_per_cell,
                                fjh::density, fjh::alpha_source>(
          resolved_pkgs.get());
    } else {
      return MakePackDescriptor<fjh::density, fjh::alpha_source>(resolved_pkgs.get());
    }
  }();
  auto vmesh = desc.GetPack(md);

  // Indexing and dimensionality
  const auto &ib = md->GetBoundsI(IndexDomain::interior);
  const auto &jb = md->GetBoundsJ(IndexDomain::interior);
  const auto &kb = md->GetBoundsK(IndexDomain::interior);
  const int &nblocks = vmesh.GetNBlocks();

  // MeshBlock dimensions/size
  const int nx1 = ib.e - ib.s + 1;
  const int nx2 = jb.e - jb.s + 1;
  const int nx3 = kb.e - kb.s + 1;
  const int num_cells = nx1 * nx2 * nx3;

  // Only early exits need a separate clear. Normal sourcing overwrites these
  // diagnostics alongside source counts in the same pass.
  const auto clear_source_diagnostics = [&]() {
    if constexpr (Diagnostics) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "ZeroSourceDiagnostics", DevExecSpace(), 0,
          nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
            vmesh(b, fj::source_num_per_cell(), k, j, i) = 0.0;
            vmesh(b, fj::source_ew_per_cell(), k, j, i) = 0.0;
            vmesh(b, fj::delta_num_per_cell(), k, j, i) = 0.0;
          });
    }
  };
  // Initialization calls use dt=0 and must not overwrite the step workspace.
  if (dt <= 0.0) {
    clear_source_diagnostics();
    return TaskStatus::complete;
  }

  auto scratch = cpt_pkg->template Param<std::shared_ptr<TransportScratch>>(
      "transport_scratch");
  scratch->ResizeSource(nblocks, num_cells);
  auto source_count = scratch->source_count;
  // An early return must leave population control with zero source counts, but
  // the normal counting kernel overwrites every cell and needs no pre-clear.
  const auto clear_source_counts = [&]() {
    clear_source_diagnostics();
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ZeroSourceCounts", DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const int cell = (k - kb.s) * nx1 * nx2 + (j - jb.s) * nx1 + i - ib.s;
          source_count(b, cell) = 0;
        });
  };

  // The cap is intentionally checked only between source calls. A single source call
  // may take the cumulative physical-alpha count past the cap; all later calls source
  // nothing. Clear counts and diagnostics only on this early-exit path.
  const Real cumulative_alphas_sourced =
      cpt_pkg->template Param<Real>("cumulative_alphas_sourced");
  if (cumulative_alphas_sourced >= alpha_source_physical_cap) {
    clear_source_counts();
    return TaskStatus::complete;
  }

  // SourceStrategy::t0 sources particles everywhere equally
  if (strategy == SourceStrategy::t0 && !Qmcblock) {
    if (t_start > 0.0) { clear_source_counts(); return TaskStatus::complete; }
  }

  const auto &mesh = md->GetMeshPointer();

  // begin pulsing logic
  const int &pulses = cpt_pkg->template Param<int>("pulses");
  const int &source_time_pulses = cpt_pkg->template Param<int>("source_time_pulses");
  const bool line_source_from_qfield =
      cpt_pkg->template Param<bool>("line_source_from_qfield");
  const Real &tlim = cpt_pkg->template Param<Real>("tlim");
  const auto &variable_field =
      cpt_pkg->template Param<std::string>("variable_field");

  int current_cycle = static_cast<int>(std::round(t_start / dt));
  const bool use_qfield_source = (strategy == SourceStrategy::qfield);
  const bool qfield_is_rate = (alpha_mode == "rate");
  const bool use_host_line_source =
      (strategy == SourceStrategy::line && line_source_from_qfield);
  const bool use_sine_test = (variable_field == "sine_test");

  int target_particles = (current_cycle == 0) ? num_particles : num_particles_per_pulse;
  Real source_birth_time = t_start;
  if (use_sine_test) {
    PARTHENON_REQUIRE(tlim > 0.0,
                      "variable_field=sine_test requires parthenon/time.tlim > 0");
    PARTHENON_REQUIRE(num_particles >= 0,
                      "variable_field=sine_test requires num_particles >= 0");
    PARTHENON_REQUIRE(
        num_particles_per_pulse >= 0,
        "variable_field=sine_test requires num_particles_per_pulse >= 0");
    target_particles = SineTestParticleCount(
        t_start, tlim, num_particles, num_particles_per_pulse);
    if (target_particles == 0) { clear_source_counts(); return TaskStatus::complete; }
  } else if (!use_qfield_source && !use_host_line_source && !Qmcblock && pulses > 0) {
    const Real source_duration = (tlim > 0.0) ? tlim : dt;
    const Real pulse_interval = source_duration / static_cast<Real>(pulses);
    PARTHENON_REQUIRE(pulse_interval > 0.0,
                      "Pulse interval must be positive. Check tlim and pulses.");

    const Real eps = 16.0 * EPS * std::max(std::abs(t_start) + std::abs(dt),
                                           pulse_interval);
    const Real t_end = t_start + dt;
    const int first_pulse =
        std::max(0, static_cast<int>(std::ceil((t_start - eps) / pulse_interval)));
    const int last_pulse =
        std::min(pulses - 1,
                 static_cast<int>(std::floor((t_end - eps) / pulse_interval)));

    if (first_pulse > last_pulse) { clear_source_counts(); return TaskStatus::complete; }

    target_particles = 0;
    for (int pulse = first_pulse; pulse <= last_pulse; ++pulse) {
      target_particles += (pulse == 0) ? num_particles : num_particles_per_pulse;
    }
    current_cycle = first_pulse;
    source_birth_time = static_cast<Real>(first_pulse) * pulse_interval;
  }

  // mbclock sourcing
  if (use_qfield_source) {
    // The host Q field controls whether and where alpha particles are born.
  } else if (Qmcblock) {
    if (alpha_mode == "total") {
      if (current_cycle > 0) {
        clear_source_counts();
        return TaskStatus::complete; // Source all at t=0
      }
    }
  }

  // Calculate the particle budget for this specific cycle. Geometry-aware sources
  // apportion this target over source-bearing blocks below.
  Real npc = 0.0;
  if (use_host_line_source) {
    npc = 0.0;
  } else if (strategy == SourceStrategy::line) {
    npc = 0.0;
  } else if (strategy == SourceStrategy::center || strategy == SourceStrategy::point) {
    npc = 0.0;
  } else if (strategy == SourceStrategy::fuel) {
    npc = 0.0;
  } else if (strategy == SourceStrategy::plane) {
      npc = 0.0;
  } else if (strategy == SourceStrategy::qfield) {
      npc = 0.0;
  } else {
      npc = std::floor(static_cast<Real>(target_particles) /
                       (num_cells * md->GetMeshPointer()->nbtotal));
  }

  parthenon::Mesh* pmesh = nullptr;
  if constexpr (std::is_same_v<T, parthenon::MeshData<Real>>) {
      pmesh = md->GetParentPointer();
  } else {
      pmesh = md->GetBlockPointer()->pmy_mesh;
  }
  const Real global_x_min = pmesh->mesh_size.xmin(parthenon::CoordinateDirection::X1DIR);
  const Real global_y_min = pmesh->mesh_size.xmin(parthenon::CoordinateDirection::X2DIR);
  const Real global_x_center =
      0.5 * (global_x_min +
             pmesh->mesh_size.xmax(parthenon::CoordinateDirection::X1DIR));
  const Real global_y_center =
      0.5 * (global_y_min +
             pmesh->mesh_size.xmax(parthenon::CoordinateDirection::X2DIR));
  const Real global_z_center =
      0.5 * (pmesh->mesh_size.xmin(parthenon::CoordinateDirection::X3DIR) +
             pmesh->mesh_size.xmax(parthenon::CoordinateDirection::X3DIR));

  if (strategy == SourceStrategy::point) {
    PARTHENON_REQUIRE(
        pmesh->mesh_size.nx(parthenon::CoordinateDirection::X1DIR) % 2 == 1 &&
            pmesh->mesh_size.nx(parthenon::CoordinateDirection::X2DIR) % 2 == 1 &&
            pmesh->mesh_size.nx(parthenon::CoordinateDirection::X3DIR) % 2 == 1,
        "source_strategy=point requires an odd number of cells in every dimension.");
  }

  // Sourcing in center for ICF
  int center_source_cells = 0;
  if (strategy == SourceStrategy::center || strategy == SourceStrategy::point) {
    parthenon::par_reduce(
        DEFAULT_LOOP_PATTERN, "CountCenterSourceCells", parthenon::DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      int &count) {
          auto &coords = vmesh.GetCoordinates(b);
          const Real xi = coords.template Xc<parthenon::X1DIR>(i);
          const Real yi = coords.template Xc<parthenon::X2DIR>(j);
          const Real zi = coords.template Xc<parthenon::X3DIR>(k);
          const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
          const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
          const Real dx_k = coords.template Dxc<parthenon::X3DIR>(k, j, i);
          const Real source_x =
              (strategy == SourceStrategy::point) ? global_x_center : 0.0;
          const Real source_y =
              (strategy == SourceStrategy::point) ? global_y_center : 0.0;
          const Real source_z =
              (strategy == SourceStrategy::point) ? global_z_center : 0.0;
          const Real abs_xi = std::abs(xi - source_x);
          const Real abs_yi = std::abs(yi - source_y);
          const Real abs_zi = std::abs(zi - source_z);
          const Real tol_i = 16.0 * EPS * ((abs_xi > dx_i) ? abs_xi : dx_i);
          const Real tol_j = 16.0 * EPS * ((abs_yi > dx_j) ? abs_yi : dx_j);
          const Real tol_k = 16.0 * EPS * ((abs_zi > dx_k) ? abs_zi : dx_k);
          const bool is_nearest_center =
              (abs_xi <= 0.5 * dx_i + tol_i) &&
              (abs_yi <= 0.5 * dx_j + tol_j) &&
              (abs_zi <= 0.5 * dx_k + tol_k);
          if (is_nearest_center) count += 1;
        },
        Kokkos::Sum<int>(center_source_cells));
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &center_source_cells, 1, MPI_INT,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif
    if (strategy == SourceStrategy::point) {
      PARTHENON_REQUIRE(
          center_source_cells == 1,
          "Point source did not find exactly one geometrically centermost cell.");
    } else {
      PARTHENON_REQUIRE(center_source_cells > 0,
                        "Center source did not find any cells nearest the origin.");
    }
  }

  auto assign_block_targets =
      [&](auto &source_measure_per_block, auto &source_particles_per_block,
          const char *label) {
        auto measure_h = source_measure_per_block.GetHostMirrorAndCopy();
        auto particles_h = source_particles_per_block.GetHostMirror();

        Real local_measure = 0.0;
        for (int b = 0; b < nblocks; ++b) {
          local_measure += measure_h(b);
        }

        Real rank_measure_offset = 0.0;
        Real total_measure = local_measure;
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Exscan(&local_measure, &rank_measure_offset, 1,
                                       MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
        if (parthenon::Globals::my_rank == 0) rank_measure_offset = 0.0;
        PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &total_measure, 1,
                                          MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
        PARTHENON_REQUIRE(
            total_measure > 0.0 || target_particles == 0,
            std::string(label) + " source did not find any positive source measure.");

        Real prefix_measure = rank_measure_offset;
        for (int b = 0; b < nblocks; ++b) {
          const Real start_measure = prefix_measure;
          const Real end_measure = start_measure + measure_h(b);
          int block_target = 0;
          if (total_measure > 0.0 && target_particles > 0) {
            const Real scaled_start =
                static_cast<Real>(target_particles) * start_measure / total_measure;
            const Real scaled_end =
                static_cast<Real>(target_particles) * end_measure / total_measure;
            block_target = static_cast<int>(std::floor(scaled_end)) -
                           static_cast<int>(std::floor(scaled_start));
          }
          particles_h(b) = block_target;
          prefix_measure = end_measure;
        }
        source_particles_per_block.DeepCopy(particles_h);
      };

  ParArray1D<int> line_cells_per_block("line source cells per block", nblocks);
  ParArray1D<int> line_particles_per_block("line source particles per block", nblocks);
  ParArray2D<int> line_cell_prefix("line source cell prefix", nblocks, num_cells);
  ParArray1D<Real> line_measure_per_block("line source measure per block", nblocks);
  int line_source_cells = 0;
  if (strategy == SourceStrategy::line) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "CountLineSourceCells", DevExecSpace(), 0, 0, 0,
        nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          int block_count = 0;
          Real block_measure = 0.0;
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, int &count) {
                const Real xi = coords.template Xc<parthenon::X1DIR>(i);
                const Real yi = coords.template Xc<parthenon::X2DIR>(j);
                const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
                const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
                const Real x_left_edge = xi - 0.5 * dx_i;
                const Real y_bottom_edge = yi - 0.5 * dx_j;
                const bool is_global_left =
                    std::abs(x_left_edge - global_x_min) < 1.0e-8;
                const bool is_global_bottom =
                    std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
                if (is_global_left && k == kb.s) { count += 1; }
              },
              Kokkos::Sum<int>(block_count));
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, Real &measure) {
                const Real xi = coords.template Xc<parthenon::X1DIR>(i);
                const Real yi = coords.template Xc<parthenon::X2DIR>(j);
                const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
                const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
                const Real dx_k = coords.template Dxc<parthenon::X3DIR>(k, j, i);
                const Real x_left_edge = xi - 0.5 * dx_i;
                const Real y_bottom_edge = yi - 0.5 * dx_j;
                const bool is_global_left =
                    std::abs(x_left_edge - global_x_min) < 1.0e-8;
                const bool is_global_bottom =
                    std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
                if (is_global_left && k == kb.s) { measure += dx_k; }
              },
              Kokkos::Sum<Real>(block_measure));
          Kokkos::single(Kokkos::PerTeam(member), [&]() {
            line_cells_per_block(b) = block_count;
            line_measure_per_block(b) = block_measure;
          });
        });
    Kokkos::fence();

    auto line_cells_per_block_h = line_cells_per_block.GetHostMirrorAndCopy();
    int local_line_source_cells = 0;
    for (int b = 0; b < nblocks; ++b) {
      local_line_source_cells += line_cells_per_block_h(b);
    }
    line_source_cells = local_line_source_cells;
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &line_source_cells, 1, MPI_INT,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif
    PARTHENON_REQUIRE(line_source_cells > 0,
                      "Line source did not find any cells on x1min/x2min.");
    assign_block_targets(line_measure_per_block, line_particles_per_block, "Line");

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "BuildLineSourceCellPrefix", DevExecSpace(), 0, 0,
        0, nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(member, num_cells),
              [&](const int &idx, int &update, const bool &finale) {
                int k = idx / (nx1 * nx2) + kb.s;
                int j = (idx / nx1) % nx2 + jb.s;
                int i = idx % nx1 + ib.s;
                const Real xi = coords.template Xc<parthenon::X1DIR>(i);
                const Real yi = coords.template Xc<parthenon::X2DIR>(j);
                const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
                const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
                const Real x_left_edge = xi - 0.5 * dx_i;
                const Real y_bottom_edge = yi - 0.5 * dx_j;
                const bool is_global_left =
                    std::abs(x_left_edge - global_x_min) < 1.0e-8;
                const bool is_global_bottom =
                    std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
                if (finale) line_cell_prefix(b, idx) = update;
                if (is_global_left && k == kb.s) { update += 1; }
              });
        });
    Kokkos::fence();
  }

  // Source particles isotropically in the fuel as defined by the input deck
  ParArray1D<int> fuel_cells_per_block("fuel source cells per block", nblocks);
  ParArray1D<int> fuel_particles_per_block("fuel source particles per block", nblocks);
  ParArray2D<int> fuel_cell_prefix("fuel source cell prefix", nblocks, num_cells);
  ParArray1D<Real> fuel_measure_per_block("fuel source measure per block", nblocks);
  int fuel_source_cells = 0;
  if (strategy == SourceStrategy::fuel) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "CountFuelSourceCells", DevExecSpace(), 0, 0, 0,
        nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          int block_count = 0;
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, int &count) {
                if (vmesh(b, fjh::density(), k, j, i) > fuel_density_threshold) {
                  count += 1;
                }
              },
              Kokkos::Sum<int>(block_count));
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { fuel_cells_per_block(b) = block_count; });
        });
    Kokkos::fence();

    auto fuel_cells_per_block_h = fuel_cells_per_block.GetHostMirrorAndCopy();
    int local_fuel_source_cells = 0;
    for (int b = 0; b < nblocks; ++b) {
      local_fuel_source_cells += fuel_cells_per_block_h(b);
    }
    fuel_source_cells = local_fuel_source_cells;
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &fuel_source_cells, 1, MPI_INT,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif
    PARTHENON_REQUIRE(fuel_source_cells > 0,
                      "Fuel source did not find any cells above fuel_density_threshold.");
    // Verbose sourcing
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "[CHARon Sourcing]"
                << " | Fuel source cells: " << fuel_source_cells
                << " | fuel_density_threshold: " << fuel_density_threshold
                << std::endl;
    }

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "MeasureFuelSourceBlocks", DevExecSpace(), 0, 0,
        0, nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          Real block_measure = 0.0;
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, Real &measure) {
                if (vmesh(b, fjh::density(), k, j, i) > fuel_density_threshold) {
                  measure += coords.CellVolume(k, j, i);
                }
              },
              Kokkos::Sum<Real>(block_measure));
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { fuel_measure_per_block(b) = block_measure; });
        });
    Kokkos::fence();
    assign_block_targets(fuel_measure_per_block, fuel_particles_per_block, "Fuel");

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "BuildFuelSourceCellPrefix", DevExecSpace(), 0, 0,
        0, nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(member, num_cells),
              [&](const int &idx, int &update, const bool &finale) {
                int k = idx / (nx1 * nx2) + kb.s;
                int j = (idx / nx1) % nx2 + jb.s;
                int i = idx % nx1 + ib.s;
                if (finale) fuel_cell_prefix(b, idx) = update;
                if (vmesh(b, fjh::density(), k, j, i) > fuel_density_threshold) {
                  update += 1;
                }
              });
        });
    Kokkos::fence();
  }

  ParArray1D<int> plane_cells_per_block("plane source cells per block", nblocks);
  ParArray1D<int> plane_particles_per_block("plane source particles per block", nblocks);
  ParArray2D<int> plane_cell_prefix("plane source cell prefix", nblocks, num_cells);
  ParArray1D<Real> plane_measure_per_block("plane source measure per block", nblocks);
  int plane_source_cells = 0;
  if (strategy == SourceStrategy::plane) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "CountPlaneSourceCells", DevExecSpace(), 0, 0, 0,
        nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          int block_count = 0;
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, int &count) {
                const Real xi = coords.template Xc<parthenon::X1DIR>(i);
                const Real yi = coords.template Xc<parthenon::X2DIR>(j);
                const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
                const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
                const Real x_left_edge = xi - 0.5 * dx_i;
                const Real y_bottom_edge = yi - 0.5 * dx_j;
                const bool is_global_left =
                    std::abs(x_left_edge - global_x_min) < 1.0e-8;
                const bool is_global_bottom =
                    std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
                if ((is_global_left || is_global_bottom) && k == kb.s) {
                  count += 1;
                }
              },
              Kokkos::Sum<int>(block_count));
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { plane_cells_per_block(b) = block_count; });
        });
    Kokkos::fence();

    auto plane_cells_per_block_h = plane_cells_per_block.GetHostMirrorAndCopy();
    int local_plane_source_cells = 0;
    for (int b = 0; b < nblocks; ++b) {
      local_plane_source_cells += plane_cells_per_block_h(b);
    }
    plane_source_cells = local_plane_source_cells;
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &plane_source_cells, 1, MPI_INT,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif
    PARTHENON_REQUIRE(plane_source_cells > 0,
                      "Plane source did not find any cells on x1min or x2min.");

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "MeasurePlaneSourceBlocks", DevExecSpace(), 0, 0,
        0, nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          Real block_measure = 0.0;
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, Real &measure) {
                const Real xi = coords.template Xc<parthenon::X1DIR>(i);
                const Real yi = coords.template Xc<parthenon::X2DIR>(j);
                const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
                const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
                const Real dx_k = coords.template Dxc<parthenon::X3DIR>(k, j, i);
                const Real x_left_edge = xi - 0.5 * dx_i;
                const Real y_bottom_edge = yi - 0.5 * dx_j;
                const bool is_global_left =
                    std::abs(x_left_edge - global_x_min) < 1.0e-8;
                const bool is_global_bottom =
                    std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
                if (k == kb.s) {
                  if (is_global_left) measure += dx_j * dx_k;
                  if (is_global_bottom) measure += dx_i * dx_k;
                }
              },
              Kokkos::Sum<Real>(block_measure));
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { plane_measure_per_block(b) = block_measure; });
        });
    Kokkos::fence();
    assign_block_targets(plane_measure_per_block, plane_particles_per_block, "Plane");

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "BuildPlaneSourceCellPrefix", DevExecSpace(), 0, 0,
        0, nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(member, num_cells),
              [&](const int &idx, int &update, const bool &finale) {
                int k = idx / (nx1 * nx2) + kb.s;
                int j = (idx / nx1) % nx2 + jb.s;
                int i = idx % nx1 + ib.s;
                const Real xi = coords.template Xc<parthenon::X1DIR>(i);
                const Real yi = coords.template Xc<parthenon::X2DIR>(j);
                const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
                const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
                const Real x_left_edge = xi - 0.5 * dx_i;
                const Real y_bottom_edge = yi - 0.5 * dx_j;
                const bool is_global_left =
                    std::abs(x_left_edge - global_x_min) < 1.0e-8;
                const bool is_global_bottom =
                    std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
                if (finale) plane_cell_prefix(b, idx) = update;
                if ((is_global_left || is_global_bottom) && k == kb.s) {
                  update += 1;
                }
              });
        });
    Kokkos::fence();
  }

  const bool use_volumetric_block_source =
      strategy == SourceStrategy::xhat || strategy == SourceStrategy::isotropic ||
      strategy == SourceStrategy::t0;
  ParArray1D<int> volumetric_particles_per_block(
      "volumetric source particles per block", nblocks);
  ParArray1D<Real> volumetric_measure_per_block(
      "volumetric source measure per block", nblocks);
  if (use_volumetric_block_source) {
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "MeasureVolumetricSourceBlocks", DevExecSpace(), 0,
        0, 0, nblocks - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
          auto &coords = vmesh.GetCoordinates(b);
          Real block_measure = 0.0;
          par_reduce_inner(
              member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](const int &k, const int &j, const int &i, Real &measure) {
                measure += coords.CellVolume(k, j, i);
              },
              Kokkos::Sum<Real>(block_measure));
          Kokkos::single(Kokkos::PerTeam(member),
                         [&]() { volumetric_measure_per_block(b) = block_measure; });
        });
    Kokkos::fence();
    assign_block_targets(volumetric_measure_per_block, volumetric_particles_per_block,
                         "Volumetric");
  }

  auto nparticles = scratch->particles_per_block;
  auto prefix_sum = scratch->source_prefix;
  auto source_cells = scratch->source_cell_list;
  auto source_cell_count = scratch->source_cell_count;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "SourceAlphasGray_1", DevExecSpace(), 0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int &b) {
        auto &coords = vmesh.GetCoordinates(b);
        const Real &dv = coords.CellVolume(0, 0, 0);
        int block_sum = 0.0;

        par_reduce_inner(
            member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            [&](const int &k, const int &j, const int &i, int &ntot) {

              // The qfield path has no geometric source tests. Avoid evaluating
              // coordinates and every generic strategy for each mesh cell.
              if (use_qfield_source) {
                const Real q = vmesh(b, fjh::alpha_source(), k, j, i);
                const Real total = qfield_is_rate ? q * dt : q;
                const int count = total > 0.0 ? static_cast<int>(std::floor(total)) : 0;
                const int cell = (k - kb.s) * nx1 * nx2 + (j - jb.s) * nx1 + i - ib.s;
                source_count(b, cell) = count;
                ntot += count;
                if constexpr (Diagnostics) {
                  vmesh(b, fj::source_num_per_cell(), k, j, i) = count;
                  vmesh(b, fj::delta_num_per_cell(), k, j, i) = count;
                  vmesh(b, fj::source_ew_per_cell(), k, j, i) =
                      count > 0 ? 1.0 / count : 0.0;
                }
                return;
              }

              const Real xi = coords.template Xc<parthenon::X1DIR>(i);
              const Real yi = coords.template Xc<parthenon::X2DIR>(j);
              const Real zi = coords.template Xc<parthenon::X3DIR>(k);
              const Real dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
              const Real dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
              const Real dx_k = coords.template Dxc<parthenon::X3DIR>(k, j, i);
              const int cell_idx_1d =
                  (k - kb.s) * (nx1 * nx2) + (j - jb.s) * nx1 + (i - ib.s);

              const Real x_left_edge = xi - 0.5 * dx_i;
              const Real y_bottom_edge = yi - 0.5 * dx_j;
              const Real source_x =
                  (strategy == SourceStrategy::point) ? global_x_center : 0.0;
              const Real source_y =
                  (strategy == SourceStrategy::point) ? global_y_center : 0.0;
              const Real source_z =
                  (strategy == SourceStrategy::point) ? global_z_center : 0.0;
              const Real abs_xi = std::abs(xi - source_x);
              const Real abs_yi = std::abs(yi - source_y);
              const Real abs_zi = std::abs(zi - source_z);
              const Real tol_i = 16.0 * EPS * ((abs_xi > dx_i) ? abs_xi : dx_i);
              const Real tol_j = 16.0 * EPS * ((abs_yi > dx_j) ? abs_yi : dx_j);
              const Real tol_k = 16.0 * EPS * ((abs_zi > dx_k) ? abs_zi : dx_k);
              const bool is_nearest_center =
                  (abs_xi <= 0.5 * dx_i + tol_i) &&
                  (abs_yi <= 0.5 * dx_j + tol_j) &&
                  (abs_zi <= 0.5 * dx_k + tol_k);

              const bool is_global_left = std::abs(x_left_edge - global_x_min) < 1.0e-8;
              const bool is_global_bottom = std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
              const bool is_fuel =
                  vmesh(b, fjh::density(), k, j, i) > fuel_density_threshold;

              bool source_here = false;
              if (strategy == SourceStrategy::line) {
                  source_here = (is_global_left && k == kb.s);
              } else if (strategy == SourceStrategy::center ||
                         strategy == SourceStrategy::point) {
                  source_here = is_nearest_center;
              } else if (strategy == SourceStrategy::plane) {
                  source_here = ((is_global_left || is_global_bottom) && k == kb.s);
              } else if (strategy == SourceStrategy::fuel) {
                  source_here = is_fuel;
              } else if (strategy == SourceStrategy::xhat ||
                         strategy == SourceStrategy::isotropic ||
                         strategy == SourceStrategy::t0 ||
                         strategy == SourceStrategy::qfield) {
                  source_here = true; // Volumetric sourcing
              }
              Real cell_npc = npc;
              if (use_volumetric_block_source && source_here) {
                  const int block_target = volumetric_particles_per_block(b);
                  cell_npc = block_target / num_cells;
                  const int remainder = block_target % num_cells;
                  if (cell_idx_1d < remainder) cell_npc += 1;
              }
              if (strategy == SourceStrategy::line && source_here) {
                  const int block_target = line_particles_per_block(b);
                  const int block_source_cells = line_cells_per_block(b);
                  cell_npc = block_source_cells > 0 ? block_target / block_source_cells : 0;
                  const int remainder =
                      block_source_cells > 0 ? block_target % block_source_cells : 0;
                  const int cell_rank = line_cell_prefix(b, cell_idx_1d);
                  if (cell_rank < remainder) cell_npc += 1;
              }
              if ((strategy == SourceStrategy::center ||
                   strategy == SourceStrategy::point) && source_here) {
                  cell_npc = target_particles / center_source_cells;
                  const int remainder = target_particles % center_source_cells;
                  const int cell_rank =
                      ((xi > 0.0) ? 1 : 0) + 2 * ((yi > 0.0) ? 1 : 0) +
                      4 * ((zi > 0.0) ? 1 : 0);
                  if (cell_rank < remainder) cell_npc += 1;
              }
              if (strategy == SourceStrategy::fuel && source_here) {
                  const int block_target = fuel_particles_per_block(b);
                  const int block_source_cells = fuel_cells_per_block(b);
                  cell_npc = block_source_cells > 0 ? block_target / block_source_cells : 0;
                  const int remainder =
                      block_source_cells > 0 ? block_target % block_source_cells : 0;
                  const int cell_rank = fuel_cell_prefix(b, cell_idx_1d);
                  if (cell_rank < remainder) cell_npc += 1;
              }
              if (strategy == SourceStrategy::plane && source_here) {
                  const int block_target = plane_particles_per_block(b);
                  const int block_source_cells = plane_cells_per_block(b);
                  cell_npc = block_source_cells > 0 ? block_target / block_source_cells : 0;
                  const int remainder =
                      block_source_cells > 0 ? block_target % block_source_cells : 0;
                  const int cell_rank = plane_cell_prefix(b, cell_idx_1d);
                  if (cell_rank < remainder) cell_npc += 1;
              }
              if (strategy == SourceStrategy::qfield || use_host_line_source) {
                  const Real q = vmesh(b, fjh::alpha_source(), k, j, i);
                  const Real q_total = qfield_is_rate ? q * dt : q;
                  const Real q_nonnegative = (q_total > 0.0) ? q_total : 0.0;
                  cell_npc = std::floor(q_nonnegative);
                  source_here = source_here && (cell_npc > 0.0);
              }

              // Compute erad from speed of sourced particle
              Real erad = CharonNull<Real>();
              Real vv = std::sqrt((2.0 * E0) / MALPHA);
              erad = E0;

              const int new_count =
                  (source_here && cell_npc > 0.0) ? static_cast<int>(cell_npc) : 0;
              source_count(b, cell_idx_1d) = new_count;
              ntot += new_count;
              if constexpr (Diagnostics) {
                vmesh(b, fj::source_num_per_cell(), k, j, i) = new_count;
                vmesh(b, fj::delta_num_per_cell(), k, j, i) = new_count;
                vmesh(b, fj::source_ew_per_cell(), k, j, i) =
                    new_count > 0 ? 1.0 / new_count : 0.0;
              }
            },
            Kokkos::Sum<int>(block_sum));

        Kokkos::single(Kokkos::PerTeam(member), [&]() { nparticles(b) = block_sum; });
        member.team_barrier();

        Kokkos::parallel_scan(Kokkos::TeamThreadRange(member, num_cells),
                              [&](const int &idx, int &update, const bool &finale) {
                                if (finale) prefix_sum(b, idx) = update;
                                update += source_count(b, idx);
                              });
        int active_cells = 0;
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(member, num_cells),
            [&](const int idx, int &update, const bool final) {
              if (source_count(b, idx) > 0) {
                if (final) source_cells(b, update) = idx;
                ++update;
              }
            }, active_cells);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          source_cell_count(b) = active_cells;
        });
      });
  auto new_contexts = scratch->new_contexts;
  auto new_contexts_h = scratch->new_contexts_host;
  auto nparticles_h = scratch->particles_per_block_host;
  nparticles_h.DeepCopy(nparticles);

  long long step_total_local = 0;

  for (int b = 0; b <= nblocks - 1; ++b) {
      step_total_local += nparticles_h(b);
      // Release indices retained by a block that no longer sources particles.
      if (nparticles_h(b) == 0) new_contexts_h(b) = NewParticlesContext();
  }
  long long step_total = step_total_local;
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &step_total, 1, MPI_LONG_LONG,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif
  const Real updated_cumulative_alphas =
      cumulative_alphas_sourced + static_cast<Real>(step_total) * a_ww;
  cpt_pkg->template UpdateParam<Real>("cumulative_alphas_sourced",
                                      updated_cumulative_alphas);

  if (parthenon::Globals::my_rank == 0 && step_total > 0) {
    std::cout << "[CHARon Sourcing]"
              << " | Sourced this stage: " << step_total
              << " | Cumulative weighted alphas: " << updated_cumulative_alphas
              << " / " << alpha_source_physical_cap << std::endl;
  }

  // All ranks participated in the source-budget collective above. Ranks with
  // no births can skip swarm allocation and the initialization kernel entirely.
  if (step_total_local == 0) return TaskStatus::complete;

  for (int b = 0; b <= nblocks - 1; ++b) {
    if (nparticles_h(b) == 0) continue;
    auto particles = md->GetSwarmData(b)->Get(alphas_swarm_name);
    new_contexts_h(b) = particles->AddEmptyParticles(nparticles_h(b));
  }
  new_contexts.DeepCopy(new_contexts_h);

  static auto pdesc_r =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z,
                              pa::time, pa::v, pa::energy, pa::weight>(
          alphas_swarm_name);
  static auto pdesc_i = MakeSwarmPackDescriptor<pa::ijk>(alphas_swarm_name);
  auto ppack_r = pdesc_r.GetPack(md);
  auto ppack_i = pdesc_i.GetPack(md);

  constexpr Real eps = 4.0e8 * EPS;
  const Real ome = 1.0 - eps;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "SourceAlphasGray_2", DevExecSpace(),
      0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, source_cell_count(b)), [&](const int idx) {
        const int cell = source_cells(b, idx);
        const int i = ib.s + cell % nx1;
        const int j = jb.s + (cell / nx1) % nx2;
        const int k = kb.s + cell / (nx1 * nx2);
        const int new_part_per_cell = source_count(b, cell);
        if (new_part_per_cell <= 0) return;

        auto &coords = vmesh.GetCoordinates(b);
        auto rng_gen = rng_pool.get_state();

        const Real &xi = coords.template Xc<parthenon::X1DIR>(i);
        const Real &yi = coords.template Xc<parthenon::X2DIR>(j);
        const Real &zi = coords.template Xc<parthenon::X3DIR>(k);
        const Real &dx_i = coords.template Dxc<parthenon::X1DIR>(k, j, i);
        const Real &dx_j = coords.template Dxc<parthenon::X2DIR>(k, j, i);
        const Real &dx_k = coords.template Dxc<parthenon::X3DIR>(k, j, i);

        // Calculate exact left/bottom edges of THIS cell
        const Real x_left_edge = xi - 0.5 * dx_i;
        const Real y_bottom_edge = yi - 0.5 * dx_j;
        const Real source_x =
            (strategy == SourceStrategy::point) ? global_x_center : 0.0;
        const Real source_y =
            (strategy == SourceStrategy::point) ? global_y_center : 0.0;
        const Real source_z =
            (strategy == SourceStrategy::point) ? global_z_center : 0.0;
        const Real abs_xi = std::abs(xi - source_x);
        const Real abs_yi = std::abs(yi - source_y);
        const Real abs_zi = std::abs(zi - source_z);
        const Real tol_i = 16.0 * EPS * ((abs_xi > dx_i) ? abs_xi : dx_i);
        const Real tol_j = 16.0 * EPS * ((abs_yi > dx_j) ? abs_yi : dx_j);
        const Real tol_k = 16.0 * EPS * ((abs_zi > dx_k) ? abs_zi : dx_k);
        const bool is_nearest_center =
            (abs_xi <= 0.5 * dx_i + tol_i) &&
            (abs_yi <= 0.5 * dx_j + tol_j) &&
            (abs_zi <= 0.5 * dx_k + tol_k);

        // Compare to global bounds
        const bool is_global_left = std::abs(x_left_edge - global_x_min) < 1.0e-8;
        const bool is_global_bottom = std::abs(y_bottom_edge - global_y_min) < 1.0e-8;
        const bool is_fuel =
            vmesh(b, fjh::density(), k, j, i) > fuel_density_threshold;
        if (strategy == SourceStrategy::line) {
            if (!(is_global_left && k == kb.s)) return;
        } else if (strategy == SourceStrategy::center ||
                   strategy == SourceStrategy::point) {
            if (!is_nearest_center) return;
        } else if (strategy == SourceStrategy::plane) {
            if (!((is_global_left || is_global_bottom) && k == kb.s)) return;
        } else if (strategy == SourceStrategy::fuel) {
            if (!is_fuel) return;
        }
        const int cell_idx_1d = (k - kb.s) * (nx1 * nx2) + (j - jb.s) * nx1 + (i - ib.s);

        [[maybe_unused]] const Real &t_startd = source_birth_time;
        [[maybe_unused]] const Real &dtd = dt;

        // Starting index and length of particles in this cell
        const int &pstart_idx = prefix_sum(b, cell_idx_1d);

        const Real vv = std::sqrt((2.0 * E0) / MALPHA);
        for (int np = pstart_idx; np < pstart_idx + new_part_per_cell; np++) {
          const int &n = new_contexts(b).GetNewParticleIndex(np);
          ppack_i(b, pa::ijk(0), n) = i;
          ppack_i(b, pa::ijk(1), n) = j;
          ppack_i(b, pa::ijk(2), n) = k;

          // sample position uniformly over cell
          Real px = xi + dx_i * ome * (rng_gen.drand() - 0.5);
          Real py = yi + dx_j * ome * (rng_gen.drand() - 0.5);
          Real pz = zi + dx_k * ome * (rng_gen.drand() - 0.5);
          if (strategy == SourceStrategy::center) {
            px = 0.0;
            py = 0.0;
            pz = 0.0;
          } else if (strategy == SourceStrategy::point) {
            px = xi;
            py = yi;
            pz = zi;
          }

          // Evaluate directions and pin boundary coordinates
          if (strategy == SourceStrategy::xhat) {
            ppack_r(b, pa::v(0), n) = vv;  // vx
            ppack_r(b, pa::v(1), n) = 0.0; // vy
            ppack_r(b, pa::v(2), n) = 0.0; // vz

          } else if (strategy == SourceStrategy::line) {
              if (is_global_left && k == kb.s) {
              //if (is_global_left && is_global_bottom && k == kb.s) {
                px = x_left_edge + 1.0e-6;
              }
              // Isotropic outward half-space source: mu = cos(theta) is uniform in [0, 1].
              // This matches an angular measure proportional to sin(theta) dtheta.
              const Real mu = rng_gen.drand();
              //const Real mu = std::sqrt(rng_gen.drand());
              const Real phi = 2.0 * M_PI * rng_gen.drand();
              const Real nu = std::sqrt(1.0 - mu * mu);

              ppack_r(b, pa::v(0), n) = vv * mu;
              ppack_r(b, pa::v(1), n) = vv * nu * std::cos(phi);
              ppack_r(b, pa::v(2), n) = vv * nu * std::sin(phi);

          } else if (strategy == SourceStrategy::plane) {
              bool emit_left = is_global_left;
              bool emit_bottom = is_global_bottom;

              if (is_global_left && is_global_bottom) {
                if (rng_gen.drand() < 0.5) emit_bottom = false;
                  else emit_left = false;
              }

              if (emit_left) {
                if (k == kb.s) px = x_left_edge + 1.0e-6;

                  const Real mu = std::sqrt(rng_gen.drand());
                  const Real nu = std::sqrt(1.0 - mu * mu);
                  const Real phi = 2.0 * M_PI * rng_gen.drand();

                  ppack_r(b, pa::v(0), n) = vv * mu;
                  ppack_r(b, pa::v(1), n) = vv * nu * std::cos(phi);
                  ppack_r(b, pa::v(2), n) = vv * nu * std::sin(phi);

              } else if (emit_bottom) {
                  if (k == kb.s) py = y_bottom_edge + 1.0e-6;

                  const Real mu = std::sqrt(rng_gen.drand());
                  const Real nu = std::sqrt(1.0 - mu * mu);
                  const Real phi = 2.0 * M_PI * rng_gen.drand();

                  ppack_r(b, pa::v(0), n) = vv * nu * std::cos(phi);
                  ppack_r(b, pa::v(1), n) = vv * mu;
                  ppack_r(b, pa::v(2), n) = vv * nu * std::sin(phi);
              }
          } else if (strategy == SourceStrategy::center ||
                      strategy == SourceStrategy::point ||
                      strategy == SourceStrategy::fuel ||
                      strategy == SourceStrategy::isotropic ||
                      strategy == SourceStrategy::t0 ||
                      strategy == SourceStrategy::qfield) {
              const Real mu = 2.0 * rng_gen.drand() - 1.0;
              const Real phi = 2.0 * M_PI * rng_gen.drand();
              const Real nu = std::sqrt(1.0 - mu * mu);

              ppack_r(b, pa::v(0), n) = vv * mu;
              ppack_r(b, pa::v(1), n) = vv * nu * std::cos(phi);
              ppack_r(b, pa::v(2), n) = vv * nu * std::sin(phi);
              //const Real theta = std::acos(2.0 * rng_gen.drand() - 1.0);
              //const Real phi = 2.0 * M_PI * rng_gen.drand();
              //const Real stheta = std::sin(theta);
              //ppack_r(b, pa::v(0), n) = vv * stheta * std::cos(phi);
              //ppack_r(b, pa::v(1), n) = vv * stheta * std::sin(phi);
              //ppack_r(b, pa::v(2), n) = vv * std::cos(theta);
          }

          // Assign final positions
          ppack_r(b, swarm_position::x(), n) = px;
          ppack_r(b, swarm_position::y(), n) = py;
          ppack_r(b, swarm_position::z(), n) = pz;

#ifndef KOKKOS_ENABLE_CUDA
          if (verbose == true) {
            printf("RANK %d: Cell (%d,%d,%d) | Pos: (%.3e, %.3e, %.3e) | Vel: (%.3e, %.3e, %.3e)\n",
              parthenon::Globals::my_rank, i, j, k,
              px, py, pz,
              ppack_r(b, pa::v(0), n), ppack_r(b, pa::v(1), n), ppack_r(b, pa::v(2), n));
          }
#endif
          ppack_r(b, pa::energy(), n) = E0;// MeV
          if constexpr (Plasma) {
            Real lab_speed2 = 0.0;
            for (int d = 0; d < 3; ++d) {
              auto &velocity = ppack_r(b, pa::v(d), n);
              velocity += vmesh(b, fj::plasma_velocity(d), k, j, i);
              lab_speed2 += velocity * velocity;
            }
            // Swarm energy is lab kinetic energy; LP recomputes relative energy.
            ppack_r(b, pa::energy(), n) = 0.5 * MALPHA * lab_speed2;
          }
          ppack_r(b, pa::weight(), n) = a_ww;
          if constexpr (Plasma) {
            ppack_r(b, pa::time(), n) = t_startd + rng_gen.drand() * dtd;
          } else if (use_host_line_source && current_cycle == 0) {
            // sine_test defines its initial population as present at t=0.
            ppack_r(b, pa::time(), n) = t_startd;
          } else {
            const int time_pulse = np % source_time_pulses;
            const Real pulse_width = dtd / static_cast<Real>(source_time_pulses);
            ppack_r(b, pa::time(), n) =
                t_startd; 
                //(static_cast<Real>(time_pulse) + rng_gen.drand()) * pulse_width;
          }
        } // end for
        rng_pool.free_state(rng_gen);
      });
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus SourceAlphas(T *md, const Real t_start, const Real dt) {
  const bool diagnostics = md->GetParentPointer()->packages.Get("charon")
                               ->template Param<bool>("cpt_diagnostics");
  if (md->GetParentPointer()->packages.Get("charon")->template Param<bool>("tn_plasma")) {
    return diagnostics ? SourceAlphasImpl<T, true, true>(md, t_start, dt)
                       : SourceAlphasImpl<T, false, true>(md, t_start, dt);
  }
  return diagnostics ? SourceAlphasImpl<T, true>(md, t_start, dt)
                     : SourceAlphasImpl<T, false>(md, t_start, dt);
}

//----------------------------------------------------------------------------------------
//! template instantiations, only take CPT and gray cases
template TaskStatus SourceAlphas<MeshData<Real>>(MeshData<Real> *md, const Real t_start, const Real dt);
template TaskStatus SourceAlphas<MeshBlockData<Real>>(MeshBlockData<Real> *md, const Real t_start, const Real dt);
} // namespace charon
