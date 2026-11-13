// (C) 2026. Triad National Security, LLC. All rights reserved.
// This file was made in part with generative AI.
#include "cpt/rz.hpp"
#include "cpt/spherical_geometry.hpp"

#include "charon.hpp"
#include "riot_utils/riot_utils.hpp"
#include "source_profiles.hpp"
#include "stopping_power.hpp"
#include "variables.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <parthenon_mpi.hpp>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

// Uniform verification and uniform/static CPU RZ coupled TN plasma adapter. Particle weights are physical full-domain weights;
// each history has exactly one rank owner and deposits once into the global RZ grid.
namespace CPT::rz {
using namespace CPT::spherical;
using namespace parthenon::driver::prelude;
namespace fj = field::charon;
namespace cb = cell_variables::cell_averaged::bulk;
namespace {
struct Particle { Vector x{}, v{}; Real time = 0, weight = 0; };
struct State {
  std::vector<Particle> particles;
  std::mt19937_64 rng;
  unsigned long long source_sequence = 0;
  std::string checkpoint_directory;
  int nr = 0, nz = 0;
  bool static_mesh = false;
  std::vector<Real> leaf_geometry;
  int last_lp_warning_cycle = -1;
  Real rmax = 0, zmin = 0, zmax = 0;
  Real deposited = 0, escaped = 0, escaped_count = 0, thermalized = 0;
  // Signed rank-local statistical changes, separate from physical transfers.
  Real roulette_count_delta = 0, roulette_energy_delta = 0;
  Vector roulette_momentum_delta{};
};
enum Background { Inner, Outer, Zlo, Zhi, Volume, Source, SourceWeight, Ur, Uz, Ne, Te, Ti, Screening, Birth, Ions };
enum Tally { Qe, Qi, Mass, Momentum, MomentumZ, Energy, Thermal, Escape, EscapeEnergy,
             StopSum, StopCount, ActiveCount, ActiveWeight, ActiveEnergy, NTally };
struct SearchNode {
  Real rlo, rhi, zlo, zhi;
  int left=-1, right=-1, cell=-1;
};
struct BackgroundData {
  int stride = Ions, nr = 0, nz = 0;
  Real rmax = 0, zmin = 0, zmax = 0;
  std::vector<Real> rows;
  std::vector<int> local_cell;
  bool leaf_mesh=false;
  std::vector<SearchNode> tree;
  int Size() const { return leaf_mesh ? rows.size()/stride : nr*nz; }
  Real Get(int cell, int field) const { return rows[cell * stride + field]; }
};
std::shared_ptr<State> GetState(Mesh *pm) {
  return pm->packages.Get("charon")->Param<std::shared_ptr<State>>("rz_state");
}
void Sum(std::vector<Real> &values) {
  PARTHENON_REQUIRE(values.size() <= INT_MAX, "RZ CPT MPI tally too large");
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(),
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
}

// Balanced bounding-box hierarchy; leaf cells retain canonical (Z,R) IDs.
int BuildSearch(BackgroundData &bg, std::vector<int> &ids, int first, int last) {
  const int node=bg.tree.size();
  const int c=ids[first];
  SearchNode box{bg.Get(c,Inner),bg.Get(c,Outer),bg.Get(c,Zlo),bg.Get(c,Zhi)};
  for(int n=first+1;n<last;++n) {
    const int s=ids[n];
    box.rlo=std::min(box.rlo,bg.Get(s,Inner)); box.rhi=std::max(box.rhi,bg.Get(s,Outer));
    box.zlo=std::min(box.zlo,bg.Get(s,Zlo)); box.zhi=std::max(box.zhi,bg.Get(s,Zhi));
  }
  bg.tree.push_back(box);
  if(last-first==1) { bg.tree[node].cell=c; return node; }
  const bool radial=box.rhi-box.rlo >= box.zhi-box.zlo;
  const int mid=first+(last-first)/2;
  std::nth_element(ids.begin()+first,ids.begin()+mid,ids.begin()+last,[&](int a,int b) {
    const Real ca=radial ? bg.Get(a,Inner)+bg.Get(a,Outer) : bg.Get(a,Zlo)+bg.Get(a,Zhi);
    const Real cb=radial ? bg.Get(b,Inner)+bg.Get(b,Outer) : bg.Get(b,Zlo)+bg.Get(b,Zhi);
    return ca==cb ? a<b : ca<cb;
  });
  const int left=BuildSearch(bg,ids,first,mid),right=BuildSearch(bg,ids,mid,last);
  bg.tree[node].left=left; bg.tree[node].right=right;
  return node;
}
int FindLeaf(const BackgroundData &bg,int node,Real r,Real z) {
  const auto &box=bg.tree[node];
  if(r<box.rlo || r>=box.rhi || z<box.zlo || z>=box.zhi) return -1;
  if(box.cell>=0) return box.cell;
  const int left=FindLeaf(bg,box.left,r,z);
  return left>=0 ? left : FindLeaf(bg,box.right,r,z);
}
bool OverlappingLeaf(const BackgroundData &bg,int node,int cell,Real eps) {
  const auto &box=bg.tree[node];
  if(std::min(box.rhi,bg.Get(cell,Outer))-std::max(box.rlo,bg.Get(cell,Inner))<=eps ||
     std::min(box.zhi,bg.Get(cell,Zhi))-std::max(box.zlo,bg.Get(cell,Zlo))<=eps) return false;
  if(box.cell>=0) return box.cell!=cell;
  return OverlappingLeaf(bg,box.left,cell,eps) || OverlappingLeaf(bg,box.right,cell,eps);
}

// Static plasma mesh only: gather actual active leaves, never synthesize a
// finest-level global raster. Sorting makes source allocation independent of MPI layout.
BackgroundData GatherLeaves(Mesh *pm,bool material_state) {
  auto state=GetState(pm); auto pkg=pm->packages.Get("charon");
  BackgroundData bg;
  bg.leaf_mesh=true; bg.nr=state->nr; bg.nz=state->nz;
  bg.rmax=state->rmax; bg.zmin=state->zmin; bg.zmax=state->zmax;
  const int ns=pkg->Param<int>("plasma_num_species");
  const int stride=Ions+2*ns; bg.stride=stride;
  const Real weight=pkg->Param<Real>("alpha_weight");
  const bool population=pkg->Param<bool>("do_population_control");
  std::vector<Real> local;
  for(int p=0;p<pm->DefaultNumPartitions();++p) {
    auto md=pm->mesh_data.GetOrAdd("base",p);
    auto geometry=riot::MakePack<cb::cpt_alpha_source>(md.get());
    const auto ib=md->GetBoundsI(IndexDomain::interior),jb=md->GetBoundsJ(IndexDomain::interior);
    const int k=md->GetBoundsK(IndexDomain::interior).s;
    const int ni=ib.e-ib.s+1,nj=jb.e-jb.s+1,n=geometry.GetNBlocks()*ni*nj;
    if(!n) continue;
    ParArray2D<Real> staging("RZ leaf background",n,stride);
    parthenon::par_for("GatherRZLeafGeometry",0,n-1,KOKKOS_LAMBDA(int c) {
      const int b=c/(ni*nj),i=ib.s+c%ni,j=jb.s+(c/ni)%nj;
      const auto &coords=geometry.GetCoordinates(b);
      for(int f=0;f<stride;++f) staging(c,f)=0;
      staging(c,Inner)=coords.template Xf<X1DIR>(k,j,i);
      staging(c,Outer)=coords.template Xf<X1DIR>(k,j,i+1);
      staging(c,Zlo)=coords.template Xf<X2DIR>(k,j,i);
      staging(c,Zhi)=coords.template Xf<X2DIR>(k,j+1,i);
      staging(c,Volume)=coords.CellVolume(k,j,i);
      staging(c,SourceWeight)=weight;
    });
    if(material_state) {
      auto v=riot::MakePack<fj::plasma_velocity,fj::plasma_electron_density,
          fj::plasma_electron_temperature,fj::plasma_ion_temperature,fj::plasma_screening,
          fj::plasma_ion_density,fj::plasma_ion_charge,fj::source_alpha_energy,
          cb::cpt_alpha_source,fj::pending_alphas>(md.get());
      parthenon::par_for("GatherRZLeafPlasma",0,n-1,KOKKOS_LAMBDA(int c) {
        const int b=c/(ni*nj),i=ib.s+c%ni,j=jb.s+(c/ni)%nj;
        staging(c,Ur)=v(b,fj::plasma_velocity(0),k,j,i);
        staging(c,Uz)=v(b,fj::plasma_velocity(1),k,j,i);
        PARTHENON_REQUIRE(v(b,fj::plasma_velocity(2),k,j,i)==0,"RZ requires zero swirl");
        staging(c,Ne)=v(b,fj::plasma_electron_density(),k,j,i);
        staging(c,Te)=v(b,fj::plasma_electron_temperature(),k,j,i);
        staging(c,Ti)=v(b,fj::plasma_ion_temperature(),k,j,i);
        staging(c,Screening)=v(b,fj::plasma_screening(),k,j,i);
        staging(c,Source)=population ? v(b,fj::pending_alphas(),k,j,i) : v(b,cb::cpt_alpha_source(),k,j,i);
        staging(c,Birth)=v(b,fj::source_alpha_energy(),k,j,i);
        for(int s=0;s<ns;++s) {
          staging(c,Ions+2*s)=v(b,fj::plasma_ion_density(s),k,j,i);
          staging(c,Ions+2*s+1)=v(b,fj::plasma_ion_charge(s),k,j,i);
        }
      });
    }
    auto h=staging.GetHostMirrorAndCopy();
    for(int c=0;c<n;++c) for(int f=0;f<stride;++f) local.push_back(h(c,f));
  }
  PARTHENON_REQUIRE(local.size()<=INT_MAX,"RZ leaf background exceeds MPI count limit");
  int count=local.size();
  std::vector<int> counts(Globals::nranks),offsets(Globals::nranks);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD));
#else
  counts[0]=count;
#endif
  long long total=0;
  for(int r=0;r<Globals::nranks;++r) {
    offsets[r]=total; total+=counts[r];
    PARTHENON_REQUIRE(total<=INT_MAX,"RZ leaf background exceeds MPI count limit");
  }
  std::vector<Real> gathered(total);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allgatherv(local.data(),count,MPI_PARTHENON_REAL,gathered.data(),
      counts.data(),offsets.data(),MPI_PARTHENON_REAL,MPI_COMM_WORLD));
#else
  gathered=local;
#endif
  const int n=total/stride;
  PARTHENON_REQUIRE(n>0 && n<=INT_MAX/int(NTally),"Invalid RZ leaf/tally count");
  std::vector<int> order(n),inverse(n); std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](int a,int b) {
    for(int f:{Zlo,Inner,Zhi,Outer}) {
      if(gathered[a*stride+f]!=gathered[b*stride+f]) return gathered[a*stride+f]<gathered[b*stride+f];
    }
    return a<b;
  });
  bg.rows.resize(total);
  std::vector<Real> geometry; geometry.reserve(4*n);
  Real sum=0,correction=0;
  const Real eps=64*std::numeric_limits<Real>::epsilon()*std::max(bg.rmax,bg.zmax-bg.zmin);
  for(int s=0;s<n;++s) {
    inverse[order[s]]=s;
    std::copy_n(gathered.begin()+order[s]*stride,stride,bg.rows.begin()+s*stride);
    const Real lo=bg.Get(s,Inner),hi=bg.Get(s,Outer),zl=bg.Get(s,Zlo),zh=bg.Get(s,Zhi);
    const Real volume=M_PI*(hi*hi-lo*lo)*(zh-zl);
    PARTHENON_REQUIRE(lo>=-eps && hi>lo && hi<=bg.rmax+eps && zl>=bg.zmin-eps &&
        zh>zl && zh<=bg.zmax+eps && std::isfinite(volume) && volume>0 &&
        std::abs(bg.Get(s,Volume)/volume-1)<1.e-9,"Invalid full-annulus RZ leaf geometry");
    geometry.insert(geometry.end(),{lo,hi,zl,zh});
    const Real y=volume-correction,t=sum+y; correction=(t-sum)-y; sum=t;
  }
  const Real expected=M_PI*bg.rmax*bg.rmax*(bg.zmax-bg.zmin);
  PARTHENON_REQUIRE(std::abs(sum/expected-1)<1.e-10,"RZ leaves do not cover the full domain");
  std::vector<int> ids(n); std::iota(ids.begin(),ids.end(),0);
  BuildSearch(bg,ids,0,n);
  for(int c=0;c<n;++c) PARTHENON_REQUIRE(!OverlappingLeaf(bg,0,c,eps),
      "RZ leaf mesh contains overlapping or covered coarse cells");
  if(state->leaf_geometry.empty()) state->leaf_geometry=geometry;
  else PARTHENON_REQUIRE(state->leaf_geometry==geometry,
      "RZ SMR leaf geometry changed; dynamic AMR/inventory remapping is not supported");
  const int first=offsets[Globals::my_rank]/stride;
  for(int l=0;l<count/stride;++l) bg.local_cell.push_back(inverse[first+l]);
  return bg;
}

// Uniform verification retains its original indexing and transport.
BackgroundData Gather(Mesh *pm, bool material_state=true) {
  auto state = GetState(pm);
  if(state->static_mesh) return GatherLeaves(pm,material_state);
  BackgroundData bg;
  bg.nr=state->nr; bg.nz=state->nz;
  bg.rmax=state->rmax; bg.zmin=state->zmin; bg.zmax=state->zmax;
  const bool plasma=pm->packages.Get("charon")->Param<bool>("tn_plasma");
  const bool population=plasma && pm->packages.Get("charon")->Param<bool>("do_population_control");
  const int ns=plasma ? pm->packages.Get("charon")->Param<int>("plasma_num_species") : 0;
  bg.stride=Ions+2*ns;
  PARTHENON_REQUIRE(static_cast<long long>(bg.nr)*bg.nz <= INT_MAX/std::max(bg.stride, int(NTally)),
                    "RZ replicated background/tally exceeds MPI count limit");
  const int stride=bg.stride;
  bg.rows.assign(bg.Size()*bg.stride,0);
  std::vector<Real> material(bg.Size()*bg.stride,0);
  const Real dr=bg.rmax/bg.nr, dz=(bg.zmax-bg.zmin)/bg.nz;
  for (int j=0;j<bg.nz;++j) for (int i=0;i<bg.nr;++i) {
    auto *row=bg.rows.data()+(j*bg.nr+i)*bg.stride;
    row[Inner]=i*dr; row[Outer]=(i+1)*dr;
    row[Zlo]=bg.zmin+j*dz; row[Zhi]=bg.zmin+(j+1)*dz;
    row[SourceWeight]=pm->packages.Get("charon")->Param<Real>("alpha_weight");
  }
  std::vector<Real> volumes(bg.Size(),0), owners(bg.Size(),0);
  for (int p=0;p<pm->DefaultNumPartitions();++p) {
    auto md=pm->mesh_data.GetOrAdd("base",p);
    auto pack=riot::MakePack<cb::cpt_alpha_source>(md.get());
    const auto ib=md->GetBoundsI(IndexDomain::interior);
    const auto jb=md->GetBoundsJ(IndexDomain::interior);
    const int k=md->GetBoundsK(IndexDomain::interior).s;
    const int ni=ib.e-ib.s+1,nj=jb.e-jb.s+1,n=pack.GetNBlocks()*ni*nj;
    if (!n) continue;
    const Real pm_weight=pm->packages.Get("charon")->Param<Real>("alpha_weight");
    ParArray2D<Real> staging("RZ geometry",n,stride);
    parthenon::par_for("GatherRZGeometry",0,n-1,KOKKOS_LAMBDA(int c) {
      const int b=c/(ni*nj),i=ib.s+c%ni,j=jb.s+(c/ni)%nj;
      const auto &coords=pack.GetCoordinates(b);
      staging(c,0)=coords.template Xc<X1DIR>(k,j,i);
      staging(c,1)=coords.template Xc<X2DIR>(k,j,i);
      staging(c,2)=coords.CellVolume(k,j,i);
    });
    if (plasma && material_state) {
      auto v = riot::MakePack<
          fj::plasma_velocity, fj::plasma_electron_density, fj::plasma_electron_temperature,
          fj::plasma_ion_temperature, fj::plasma_screening, fj::plasma_ion_density,
          fj::plasma_ion_charge, fj::source_alpha_energy, cb::cpt_alpha_source,
          fj::pending_alphas>(md.get());
      parthenon::par_for(
          "GatherRZPlasma", 0, n - 1, KOKKOS_LAMBDA(int n) {
            const int b = n/(ni*nj), i = ib.s+n%ni, j = jb.s+(n/ni)%nj;
            staging(n, Ur) = v(b, fj::plasma_velocity(0), k, j, i);
            staging(n, Uz) = v(b, fj::plasma_velocity(1), k, j, i);
            PARTHENON_REQUIRE(v(b, fj::plasma_velocity(2), k,j,i)==0, "RZ no-swirl requires zero phi velocity");
            staging(n, Ne) = v(b, fj::plasma_electron_density(), k, j, i);
            staging(n, Te) = v(b, fj::plasma_electron_temperature(), k, j, i);
            staging(n, Ti) = v(b, fj::plasma_ion_temperature(), k, j, i);
            staging(n, Screening) = v(b, fj::plasma_screening(), k, j, i);
            staging(n, Source) = population ? v(b, fj::pending_alphas(), k,j,i)
                                          : v(b, cb::cpt_alpha_source(), k,j,i);
            staging(n, Birth) = v(b, fj::source_alpha_energy(), k, j, i);
            staging(n, SourceWeight) = pm_weight;
            for (int s = 0; s < ns; ++s) {
              staging(n, Ions + 2 * s) = v(b, fj::plasma_ion_density(s), k, j, i);
              staging(n, Ions + 2 * s + 1) = v(b, fj::plasma_ion_charge(s), k, j, i);
            }
          });
    }
    auto h=staging.GetHostMirrorAndCopy();
    for (int c=0;c<n;++c) {
      const int i=std::floor(h(c,0)/dr),j=std::floor((h(c,1)-bg.zmin)/dz);
      PARTHENON_REQUIRE(i>=0 && i<bg.nr && j>=0 && j<bg.nz,"Invalid RZ leaf cell");
      const int cell=j*bg.nr+i;
      bg.local_cell.push_back(cell);
      volumes[cell]+=h(c,2); owners[cell]+=1;
      if (plasma && material_state) for(int f=Source;f<stride;++f) material[cell*stride+f]=h(c,f);
    }
  }
  Sum(volumes); Sum(owners);
  if (plasma && material_state) {
    Sum(material);
    for(int c=0;c<bg.Size();++c) for(int f=Source;f<stride;++f)
      bg.rows[c*stride+f]=material[c*stride+f];
  }
  for (int c=0;c<bg.Size();++c) {
    const Real expected=M_PI*((bg.Get(c,Outer)*bg.Get(c,Outer))-(bg.Get(c,Inner)*bg.Get(c,Inner)))*dz;
    PARTHENON_REQUIRE(owners[c]==1 && std::abs(volumes[c]/expected-1)<1.e-9,
                      "RZ CPT requires uniform cells with full-annulus volumes");
    bg.rows[c*bg.stride+Volume]=volumes[c];
  }
  return bg;
}
int Locate(const BackgroundData &bg,const Vector &x,const Vector *velocity=nullptr) {
  if(bg.leaf_mesh) {
    Real r=std::hypot(x[0],x[1]),z=x[2];
    if(velocity) {
      // Direction selects ownership on a fine-cell face that subdivides the
      // tangential extent of a coarse neighbor, including simultaneous corners.
      const Real radial=x[0]*(*velocity)[0]+x[1]*(*velocity)[1];
      const Real inf=std::numeric_limits<Real>::infinity();
      if(radial!=0) r=std::nextafter(r,radial>0 ? inf : -inf);
      if((*velocity)[2]!=0) z=std::nextafter(z,(*velocity)[2]>0 ? inf : -inf);
    }
    const int cell=FindLeaf(bg,0,r,z);
    PARTHENON_REQUIRE(cell>=0,"RZ particle outside static leaf mesh");
    return cell;
  }
  const int i=std::floor(std::hypot(x[0],x[1])/bg.rmax*bg.nr);
  const int j=std::floor((x[2]-bg.zmin)/(bg.zmax-bg.zmin)*bg.nz);
  PARTHENON_REQUIRE(i>=0 && i<bg.nr && j>=0 && j<bg.nz,"RZ particle outside mesh");
  return j*bg.nr+i;
}
Real Uniform(State &state) { return std::generate_canonical<Real, 53>(state.rng); }
Vector Isotropic(State &state) {
  const Real mu = 2 * Uniform(state) - 1, phi = 2 * M_PI * Uniform(state);
  const Real transverse = std::sqrt(std::max(Real(0), 1 - mu * mu));
  return {transverse * std::cos(phi), transverse * std::sin(phi), mu};
}

Vector FluidVelocity(const BackgroundData &bg,int s,const Vector &x) {
  const Real r=std::hypot(x[0],x[1]);
  // Axis has no preferred radial direction; regular radial velocity vanishes there.
  const Real center=0.5*(bg.Get(s,Inner)+bg.Get(s,Outer));
  const Real scale=bg.Get(s,Inner)==0 ? bg.Get(s,Ur)/std::max(r,center)
                                    : (r>0 ? bg.Get(s,Ur)/r : 0);
  // Linear continuation through the first annulus makes u_R vanish at the axis.
  return {scale*x[0],scale*x[1],bg.Get(s,Uz)};
}
Real CellCrossing(const BackgroundData &bg,int s,const Vector &x,const Vector &d) {
  Vector tx{x[0],x[1],0},td{d[0],d[1],0};
  Real wall=std::min(SphereCrossing(tx,td,bg.Get(s,Inner)),
                     SphereCrossing(tx,td,bg.Get(s,Outer)));
  if(d[2]<0) wall=std::min(wall,(bg.Get(s,Zlo)-x[2])/d[2]);
  if(d[2]>0) wall=std::min(wall,(bg.Get(s,Zhi)-x[2])/d[2]);
  return wall;
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
      const Real r=bg.rmax*std::sqrt(Uniform(*state)),phi=2*M_PI*Uniform(*state);
      a.x={r*std::cos(phi),r*std::sin(phi),bg.zmin+(bg.zmax-bg.zmin)*Uniform(*state)};
    } // Point particles are born exactly at the origin.
    const auto direction = Isotropic(*state);
    for (int d = 0; d < 3; ++d) a.v[d] = speed * direction[d];
    a.time = start; // Same timestep-start births as non-plasma Cartesian sourcing.
    a.weight = weight;
    state->particles.push_back(a);
    shell_counts[Locate(bg, a.x, &a.v)] += 1;
  }
  state->source_sequence += count;
  Sum(shell_counts);
  for (int shell = 0; shell < bg.Size(); ++shell)
    bg.rows[shell * bg.stride + Source] = shell_counts[shell];
  pkg->UpdateParam<Real>("cumulative_alphas_sourced",
                         pkg->Param<Real>("cumulative_alphas_sourced") + count * weight);
  if (Globals::my_rank == 0 && count > 0) {
    std::cout << "[Charon Sourcing]"
              << " | Sourced this stage: " << count
              << " | Cumulative weighted alphas: "
              << pkg->Param<Real>("cumulative_alphas_sourced")
              << " / " << pkg->Param<Real>("alpha_source_physical_cap") << std::endl;
  }
}

// Straight Cartesian characteristics cross cylinders and axial planes.
// R=0 is not a wall. Face nudges disambiguate ownership without changing energy.
void TransportVerification(Mesh *pm,const BackgroundData &bg,Real end,
                           std::vector<Real> &tallies) {
  auto pkg=pm->packages.Get("charon"); auto state=GetState(pm);
  const Real stopping=pkg->Param<Real>("stopping_power_value"),mass=charon::units::MALPHA;
  const bool reflect=pkg->Param<bool>("verification_reflect_outer");
  const Real dr=bg.rmax/bg.nr,dz=(bg.zmax-bg.zmin)/bg.nz;
  std::size_t survivor=0;
  for (auto a:state->particles) {
    bool removed=false; int iterations=0;
    while (a.time<end) {
      PARTHENON_REQUIRE(++iterations<=pkg->Param<int>("max_transport_iterations"),
                        "RZ verification exceeded max_transport_iterations");
      int cell=Locate(bg,a.x);
      // A source exactly on an axial face belongs to the outgoing cell.
      const Real axial=(a.x[2]-bg.zmin)/dz;
      const int axial_face=std::round(axial);
      if (std::abs(axial-axial_face)<16*std::numeric_limits<Real>::epsilon()*bg.nz) {
        const int outgoing=axial_face-(a.v[2]<0 ? 1 : 0);
        if (outgoing>=0 && outgoing<bg.nz) cell=outgoing*bg.nr+cell%bg.nr;
      }
      const int i=cell%bg.nr,j=cell/bg.nr;
      auto *t=tallies.data()+cell*NTally;
      const Real speed=Radius(a.v);
      if (speed==0) { removed=true; break; }
      Vector direction=a.v; for (auto &v:direction) v/=speed;
      Vector tx{a.x[0],a.x[1],0},td{direction[0],direction[1],0};
      Real wall=std::numeric_limits<Real>::infinity(); int face=-1;
      auto candidate=[&](Real distance,int f) {
        if (distance>0 && distance<wall) { wall=distance; face=f; }
      };
      candidate(SphereCrossing(tx,td,bg.Get(cell,Inner)),0);
      candidate(SphereCrossing(tx,td,bg.Get(cell,Outer)),1);
      if (direction[2]<0) candidate((bg.Get(cell,Zlo)-a.x[2])/direction[2],2);
      if (direction[2]>0) candidate((bg.Get(cell,Zhi)-a.x[2])/direction[2],3);
      const auto step=ConstantDragSegment(speed,mass,stopping,end-a.time,wall);
      for (int d=0;d<3;++d) {
        a.x[d]+=direction[d]*step.distance; a.v[d]=direction[d]*step.speed;
      }
      a.time+=step.dt;
      t[Qe]+=a.weight*step.loss; state->deposited+=a.weight*step.loss;
      t[StopSum]+=stopping; t[StopCount]+=1;
      if (step.speed==0) { state->thermalized+=a.weight; removed=true; break; }
      if (!step.hit_wall) { a.time=end; continue; }
      const bool outer=(face==1 && i==bg.nr-1) || (face==2 && j==0) || (face==3 && j==bg.nz-1);
      if (outer && !reflect) {
        state->escaped+=a.weight*0.5*mass*(step.speed*step.speed);
        state->escaped_count+=a.weight; removed=true; break;
      }
      if (face<2) {
        const Real r=std::hypot(a.x[0],a.x[1]);
        if (outer) {
          const Real p=(a.x[0]*a.v[0]+a.x[1]*a.v[1])/(r*r);
          a.v[0]-=2*p*a.x[0]; a.v[1]-=2*p*a.x[1];
        }
        const Real target=(face==0 ? bg.Get(cell,Inner) : bg.Get(cell,Outer))+
                          ((face==0 || outer) ? -1 : 1)*1.e-10*dr;
        a.x[0]*=target/r; a.x[1]*=target/r;
      } else {
        if (outer) a.v[2]=-a.v[2];
        a.x[2]=(face==2 ? bg.Get(cell,Zlo) : bg.Get(cell,Zhi))+
               (face==2 ? -1 : 1)*(outer ? -1 : 1)*1.e-10*dz;
      }
      if (face>=2) {
        const Real r=std::hypot(a.x[0],a.x[1]);
        const Real radial_direction=a.x[0]*direction[0]+a.x[1]*direction[1];
        const Real rface=radial_direction>0 ? bg.Get(cell,Outer) : bg.Get(cell,Inner);
        if (r>0 && rface>0 && std::abs(r-rface)<1.e-12*dr) {
          const Real target=rface+(radial_direction>0 ? 1 : -1)*1.e-10*dr;
          a.x[0]*=target/r; a.x[1]*=target/r;
        }
      }
      if (face<2 && direction[2]!=0) {
        const Real zface=direction[2]>0 ? bg.Get(cell,Zhi) : bg.Get(cell,Zlo);
        if (std::abs(a.x[2]-zface)<1.e-12*dz)
          a.x[2]=zface+(direction[2]>0 ? 1 : -1)*1.e-10*dz;
      }
      // Resolve simultaneous outer-cylinder/endcap hits as a single event.
      const Real r=std::hypot(a.x[0],a.x[1]);
      if (r>=bg.rmax || a.x[2]<=bg.zmin || a.x[2]>=bg.zmax) {
        if (!reflect) {
          state->escaped+=a.weight*0.5*mass*(step.speed*step.speed);
          state->escaped_count+=a.weight; removed=true; break;
        }
        if (r>=bg.rmax) {
          const Real p=(a.x[0]*a.v[0]+a.x[1]*a.v[1])/(r*r);
          a.v[0]-=2*p*a.x[0]; a.v[1]-=2*p*a.x[1];
          a.x[0]*=(bg.rmax-1.e-10*dr)/r; a.x[1]*=(bg.rmax-1.e-10*dr)/r;
        }
        if (a.x[2]<=bg.zmin) { a.x[2]=bg.zmin+1.e-10*dz; a.v[2]=std::abs(a.v[2]); }
        if (a.x[2]>=bg.zmax) { a.x[2]=bg.zmax-1.e-10*dz; a.v[2]=-std::abs(a.v[2]); }
      }
    }
    if (!removed) { a.time=end; state->particles[survivor++]=a; }
  }
  state->particles.resize(survivor);
}
// Match the spherical largest-remainder source budget on RZ cell inventories.
void AllocateSourcePopulation(Mesh *pm, BackgroundData &bg) {
  auto pkg = pm->packages.Get("charon");
  const int target = pkg->Param<int>("target_sourcing_population");
  std::vector<int> order;
  for (int s = 0; s < bg.Size(); ++s) {
    const Real q = bg.Get(s, Source);
    PARTHENON_REQUIRE(std::isfinite(q) && q >= 0, "Invalid pending alpha population");
    if (q > 0) order.push_back(s);
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return bg.Get(a, Source) > bg.Get(b, Source);
  });
  if (order.size() > static_cast<std::size_t>(target)) order.resize(target);
  Real total = 0;
  for (int s : order) total += bg.Get(s, Source);
  PARTHENON_REQUIRE(std::isfinite(total), "Nonfinite controlled source inventory");
  std::vector<int> counts(bg.Size(), 0);
  if (!order.empty()) {
    const int budget = std::max(static_cast<int>(order.size()),
        static_cast<int>(std::ceil(std::min(total, Real(target)))));
    const int extra = budget - order.size();
    std::vector<std::pair<Real, int>> remainder;
    int used = 0;
    for (int s : order) {
      const Real allocation = extra * (bg.Get(s, Source) / total);
      const int n = static_cast<int>(std::floor(allocation));
      counts[s] = 1 + n;
      used += counts[s];
      remainder.emplace_back(allocation - n, s);
    }
    std::stable_sort(remainder.begin(), remainder.end(), [](const auto &a, const auto &b) {
      return a.first > b.first;
    });
    PARTHENON_REQUIRE(budget-used >= 0 && budget-used <= remainder.size(),
                      "Invalid source budget apportionment");
    for (int n = 0; n < budget-used; ++n) ++counts[remainder[n].second];
  }
  for (int s = 0; s < bg.Size(); ++s) {
    const Real physical = bg.Get(s, Source);
    bg.rows[s*bg.stride+Source] = counts[s];
    bg.rows[s*bg.stride+SourceWeight] = counts[s] ? physical / counts[s] : 0;
  }
}

// Commit actual emitted weight and residual inventory to native fields. This
// keeps uniform RZ restart accounting independent of the fixed alpha_weight.
void CommitControlledSource(Mesh *pm, const BackgroundData &bg) {
  std::size_t local = 0;
  for (int p = 0; p < pm->DefaultNumPartitions(); ++p) {
    auto md = pm->mesh_data.GetOrAdd("base", p);
    auto v = riot::MakePack<cb::cpt_alpha_source, fj::pending_alphas,
                           fj::pending_alpha_energy, fj::sampled_alphas>(md.get());
    const auto ib = md->GetBoundsI(IndexDomain::interior);
    const int k = md->GetBoundsK(IndexDomain::interior).s;
    const auto jb = md->GetBoundsJ(IndexDomain::interior);
    const int nj = jb.e-jb.s+1;
    const int ni = ib.e-ib.s+1, n = v.GetNBlocks()*ni*nj;
    if (!n) continue;
    ParArray2D<Real> data("Controlled alpha source", n, 2);
    auto h = data.GetHostMirror();
    for (int c = 0; c < n; ++c) {
      const int s = bg.local_cell.at(local++);
      h(c, 0) = bg.Get(s, Source);
      h(c, 1) = bg.Get(s, Source) * bg.Get(s, SourceWeight);
    }
    data.DeepCopy(h);
    parthenon::par_for("CommitControlledAlphaSource", 0, n-1, KOKKOS_LAMBDA(int c) {
      const int b = c/(ni*nj), i = ib.s+c%ni, j = jb.s+(c/ni)%nj;
      v(b, cb::cpt_alpha_source(), k,j,i) = data(c,0);
      if (data(c,0) > 0) {
        v(b, fj::sampled_alphas(), k,j,i) += data(c,1);
        v(b, fj::pending_alphas(), k,j,i) = 0;
        v(b, fj::pending_alpha_energy(), k,j,i) = 0;
      }
    });
  }
}

void SourceParticles(Mesh *pm, BackgroundData &bg, Real start, Real dt) {
  if (!pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
    SourceVerification(pm, bg, start, dt);
    return;
  }
  auto state = GetState(pm);
  auto pkg = pm->packages.Get("charon");
  const bool population = pkg->Param<bool>("do_population_control");
  if (population) AllocateSourcePopulation(pm, bg);
  const unsigned long long nranks = Globals::nranks, rank = Globals::my_rank;
  unsigned long long stage_histories = 0;
  for (int s = 0; s < bg.Size(); ++s) {
    const Real source = bg.Get(s, Source);
    PARTHENON_REQUIRE(std::isfinite(source) && source >= 0 && source <= INT_MAX &&
                          source == std::floor(source),
                      "Invalid radial source count");
    const unsigned long long count = source;
    stage_histories += count;
    const Real weight = bg.Get(s, SourceWeight);
    PARTHENON_REQUIRE(!count || (std::isfinite(weight) && weight>0), "Invalid RZ source weight");
    const auto first = FirstSourceForRank(state->source_sequence, rank, nranks);
    for (auto n = first; n < count; n += nranks) {
      Particle a;
      const Real lo=bg.Get(s,Inner),hi=bg.Get(s,Outer);
      Real r=std::sqrt(lo*lo+Uniform(*state)*(hi*hi-lo*lo));
      Real z=bg.Get(s,Zlo)+Uniform(*state)*(bg.Get(s,Zhi)-bg.Get(s,Zlo));
      // Only disambiguate endpoint/roundoff births; otherwise retain uniform sampling.
      const Real er=1.e-10*(hi-lo),ez=1.e-10*(bg.Get(s,Zhi)-bg.Get(s,Zlo));
      if(r<=lo) r=lo+er;
      if(r>=hi) r=hi-er;
      if(z<=bg.Get(s,Zlo)) z=bg.Get(s,Zlo)+ez;
      if(z>=bg.Get(s,Zhi)) z=bg.Get(s,Zhi)-ez;
      const Real phi=2*M_PI*Uniform(*state);
      a.x={r*std::cos(phi),r*std::sin(phi),z};
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

  if (population) CommitControlledSource(pm, bg);
  Real sampled = 0;
  for (int s = 0; s < bg.Size(); ++s)
    sampled += bg.Get(s, Source) * bg.Get(s, SourceWeight);
  pkg->UpdateParam<Real>("cumulative_alphas_sourced",
                         pkg->Param<Real>("cumulative_alphas_sourced") + sampled);
  // Background source counts are already global and replicated. Do not sum
  // them over ranks again. Match the Cartesian source log before transport.
  if (Globals::my_rank == 0 && stage_histories > 0) {
    std::cout << "[Charon Sourcing] | Geometry: rz"
              << " | Time: " << start << " | Sourced this stage: " << stage_histories
              << " | Cumulative weighted alphas: "
              << pkg->Param<Real>("cumulative_alphas_sourced") << std::endl;
  }
}

void RouletteActiveCensus(Mesh *pm, int cycle) {
  auto pkg = pm->packages.Get("charon");
  if (!pkg->Param<bool>("tn_plasma") || !pkg->Param<bool>("do_population_control")) return;
  const Real kill = pkg->Param<Real>("p_roulette")/100;
  if (kill == 0) return;
  auto state = GetState(pm);
  std::vector<Real> stats(7, 0); // before, killed, signed count, lab energy, momentum
  stats[0] = state->particles.size();
  std::size_t survivor = 0;
  for (auto a : state->particles) {
    const Real old = a.weight;
    const bool keep = Uniform(*state) >= kill;
    a.weight = keep ? old/(1-kill) : 0;
    PARTHENON_REQUIRE(std::isfinite(a.weight), "Roulette survivor weight overflow");
    const Real delta = a.weight-old;
    stats[1] += !keep;
    stats[2] += delta;
    stats[3] += delta * 0.5 * charon::units::MALPHA * Dot(a.v,a.v);
    for (int d = 0; d < 3; ++d) stats[4+d] += delta * charon::units::MALPHA * a.v[d];
    if (keep) state->particles[survivor++] = a;
  }
  state->particles.resize(survivor);
  state->roulette_count_delta += stats[2];
  state->roulette_energy_delta += stats[3];
  for (int d = 0; d < 3; ++d) state->roulette_momentum_delta[d] += stats[4+d];
  Sum(stats);
  std::vector<Real> cumulative{state->roulette_count_delta, state->roulette_energy_delta,
      state->roulette_momentum_delta[0], state->roulette_momentum_delta[1],
      state->roulette_momentum_delta[2]};
  Sum(cumulative);
  if (Globals::my_rank == 0 && stats[0] > 0)
    std::cout << "[Charon Roulette] geometry=rz cycle=" << cycle << " active_before=" << stats[0]
              << " killed=" << stats[1] << " count_delta=" << stats[2]
              << " energy_delta_erg=" << stats[3]
              << " cumulative_count_delta=" << cumulative[0]
              << " cumulative_energy_delta_erg=" << cumulative[1]
              << " cumulative_momentum_delta_g_cm_s=" << cumulative[2] << ","
              << cumulative[3] << "," << cumulative[4] << std::endl;
}

void Transport(Mesh *pm, const BackgroundData &bg, Real end, std::vector<Real> &tallies,
               int &expanded_log_used) {
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
      PARTHENON_REQUIRE(++steps <= limit, "RZ CPT substep limit exceeded");
      const int s = Locate(bg, a.x, &a.v);
      PARTHENON_REQUIRE(s < bg.Size(), "Census particle outside RZ domain");
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
                                                 Globals::my_rank, s, -1, &expanded_log_used};
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
        const Real width = std::min(bg.Get(s, Outer) - bg.Get(s, Inner), bg.Get(s,Zhi)-bg.Get(s,Zlo));
        const Real lab_bound = speed + Radius(u);
        if (lab_bound > 0) {
          h = std::min(h, 0.25 * width / lab_bound);
          if (Radius(u) > 0)
            h = std::min(h,
                         std::sqrt(tolerance) * std::max(std::hypot(a.x[0],a.x[1]), width) / lab_bound);
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
              CellCrossing(bg,s,a.x,displacement);
          if (crossing >= 1 - 1.e-11) {
            accepted = true;
            break;
          }
          h *= crossing;
        }
        PARTHENON_REQUIRE(accepted && std::isfinite(h) && h > 0,
                          "RZ CPT face integration failed");
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
      // Project the Cartesian impulse at the segment midpoint into R and Z.
      Vector midpoint{};
      for (int d = 0; d < 3; ++d)
        midpoint[d] = 0.5 * (a.x[d] + old_x[d]);
      t[Qe] += a.weight * qe;
      t[Qi] += a.weight * qi;
      t[Mass] += transfer.mass;
      t[Energy] += transfer.energy;
      const Real midpoint_radius=std::hypot(midpoint[0],midpoint[1]);
      if(midpoint_radius>0)
        t[Momentum]+=(transfer.momentum[0]*midpoint[0]+transfer.momentum[1]*midpoint[1])/midpoint_radius;
      t[MomentumZ]+=transfer.momentum[2];
      // Suppress phi impulse while retaining the entire lab energy transfer.
      // Its unresolved kinetic contribution remains in matter internal energy.
      if (thermal) {
        t[Thermal] += a.weight;
        removed = true;
        break;
      }
      const Real radius=std::hypot(a.x[0],a.x[1]);
      const Real radial=a.x[0]*a.v[0]+a.x[1]*a.v[1];
      const Real er=1.e-10*(bg.Get(s,Outer)-bg.Get(s,Inner));
      const Real ez=1.e-10*(bg.Get(s,Zhi)-bg.Get(s,Zlo));
      Real target=radius;
      if(radial>0 && bg.Get(s,Outer)-radius<=er) target=bg.Get(s,Outer)+er;
      if(radial<0 && bg.Get(s,Inner)>0 && radius-bg.Get(s,Inner)<=er) target=bg.Get(s,Inner)-er;
      if(radius>0) { a.x[0]*=target/radius; a.x[1]*=target/radius; }
      if(a.v[2]>0 && bg.Get(s,Zhi)-a.x[2]<=ez) a.x[2]=bg.Get(s,Zhi)+ez;
      if(a.v[2]<0 && a.x[2]-bg.Get(s,Zlo)<=ez) a.x[2]=bg.Get(s,Zlo)-ez;
      if(target>=bg.rmax || a.x[2]<=bg.zmin || a.x[2]>=bg.zmax) {
        t[Escape]+=a.weight; t[EscapeEnergy]+=a.weight*final_energy;
        removed=true; break;
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
    const int s = Locate(bg, a.x, &a.v);
    PARTHENON_REQUIRE(s < bg.Size(), "Invalid RZ census position");
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
    const auto jb = md->GetBoundsJ(IndexDomain::interior);
    const int nj = jb.e-jb.s+1;
    const int k = md->GetBoundsK(IndexDomain::interior).s;
    const int ni = ib.e - ib.s + 1, nb = geometry.GetNBlocks();
    if (nb == 0) continue;
    ParArray2D<Real> staging("RZ CPT reduced tallies", nb * ni * nj, NTally + 2);
    auto h = staging.GetHostMirror();
    for (int n = 0; n < nb * ni * nj; ++n) {
      const int s = bg.local_cell.at(local++);
      for (int f = 0; f < NTally; ++f)
        h(n, f) = tallies[s * NTally + f];
      h(n, NTally) = bg.Get(s, Source);
      h(n, NTally+1) = bg.Get(s, Source) * bg.Get(s, SourceWeight);
    }
    staging.DeepCopy(h);
    if (!active_only && pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
      auto v =
          riot::MakePack<fj::electron_heat_step, fj::ion_heat_step, fj::deposited_mass_step,
                       fj::deposited_momentum_step, fj::deposited_lab_energy_step,
                       fj::electron_heat, fj::ion_heat, fj::thermalized_alphas,
                       fj::escaped_alphas, fj::escaped_energy>(md.get());
      parthenon::par_for(
          "ScatterRZPlasma", 0, nb * ni * nj - 1, KOKKOS_LAMBDA(int n) {
            const int b = n/(ni*nj), i = ib.s+n%ni, j = jb.s+(n/ni)%nj;
            v(b, fj::electron_heat_step(), k, j, i) = staging(n, Qe);
            v(b, fj::ion_heat_step(), k, j, i) = staging(n, Qi);
            v(b, fj::deposited_mass_step(), k, j, i) = staging(n, Mass);
            v(b, fj::deposited_lab_energy_step(), k, j, i) = staging(n, Energy);
            for (int d = 0; d < 3; ++d)
              v(b, fj::deposited_momentum_step(d), k, j, i) =
                  d == 0 ? staging(n, Momentum) : (d == 1 ? staging(n, MomentumZ) : 0);
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
      parthenon::par_for(
          "RZCPTDiagnostics", 0, nb * ni * nj - 1, KOKKOS_LAMBDA(int n) {
            const int b = n/(ni*nj), i = ib.s+n%ni, j = jb.s+(n/ni)%nj;
            diag(b, fj::active_num_per_cell(), k, j, i) = staging(n, ActiveCount);
            diag(b, fj::active_ew_per_cell(), k, j, i) = staging(n, ActiveWeight);
            diag(b, fj::active_ke_tally(), k, j, i) = staging(n, ActiveEnergy);
            if (!active_only) {
              diag(b, fj::source_num_per_cell(), k, j, i) = staging(n, NTally);
              diag(b, fj::source_ew_per_cell(), k, j, i) = staging(n, NTally+1);
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
        parthenon::par_for("RZVerificationHeat", 0, nb * ni * nj - 1,
            KOKKOS_LAMBDA(int n) {
              heat(n/(ni*nj), fj::deposited_energy_step(), k, jb.s+(n/ni)%nj, ib.s+n%ni) =
                  staging(n, Qe) + staging(n, Qi);
            });
      }
    }
    if constexpr (!Compact) {
      if (!active_only) {
        auto heat = riot::MakePack<fj::energy_tally, fj::energy_tally_step,
                                   fj::energy_delta_step>(md.get());
        parthenon::par_for(
            "RZCPTHeat", 0, nb * ni * nj - 1, KOKKOS_LAMBDA(int n) {
              const int b = n/(ni*nj), i = ib.s+n%ni, j = jb.s+(n/ni)%nj;
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
  name << "cpt-rz." << std::setw(10) << std::setfill('0') << cycle << ".rank"
       << std::setw(6) << Globals::my_rank << ".txt";
  return std::filesystem::path(state.checkpoint_directory) / name.str();
}
std::string PlasmaSignature(Mesh *pm, bool include_population=true) {
  auto pkg=pm->packages.Get("charon");
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<Real>::max_digits10)
      << pkg->Param<std::string>("stopping_power_model") << ' '
      << pkg->Param<bool>("tn_depletion") << ' ' << pkg->Param<bool>("do_feedback") << ' '
      << pkg->Param<Real>("stopping_power_value") << ' '
      << pkg->Param<Real>("constant_ion_fraction") << ' '
      << pkg->Param<Real>("plasma_thermalization_energy") << ' '
      << pkg->Param<Real>("plasma_velocity_tolerance") << ' '
      << pkg->Param<Real>("plasma_fractional_step");
  // Param returns a const reference; this Parthenon version's mirror method
  // requires a non-const array handle. Copy handles, not the package arrays.
  auto species_mass=pkg->Param<ParArray1D<Real>>("plasma_species_mass");
  auto species_charge=pkg->Param<ParArray1D<Real>>("plasma_species_nuclear_charge");
  const auto mass=species_mass.GetHostMirrorAndCopy();
  const auto charge=species_charge.GetHostMirrorAndCopy();
  for(int s=0;s<pkg->Param<int>("plasma_num_species");++s) out << ' ' << mass(s) << ' ' << charge(s);
  if(include_population)
    out << ' ' << pkg->Param<bool>("do_population_control")
        << ' ' << pkg->Param<int>("target_sourcing_population")
        << ' ' << pkg->Param<Real>("p_roulette");
  return out.str();
}

void Save(Mesh *pm, const SimTime &tm) {
  auto state = GetState(pm);
  const auto target = CheckpointPath(*state, tm.ncycle);
  const auto temporary = target.string() + ".tmp";
  std::ofstream out(temporary);
  out << std::setprecision(std::numeric_limits<Real>::max_digits10);
  const bool plasma=pm->packages.Get("charon")->Param<bool>("tn_plasma");
  out << (plasma ? (state->static_mesh ? "RIOT_RZ_PLASMA_V4 " : "RIOT_RZ_PLASMA_V3 ") : "RIOT_RZ_CPT_V1 ") << Globals::nranks << ' ' << Globals::my_rank << ' '
      << tm.ncycle << ' ' << tm.time << ' ' << state->source_sequence << ' '
      << state->particles.size() << '\n'
      << state->rng << '\n';
  if(plasma) {
    auto pkg=pm->packages.Get("charon");
    out << state->nr << ' ' << state->nz << ' ' << state->rmax << ' '
        << state->zmin << ' ' << state->zmax << ' '
        << pkg->Param<int>("plasma_num_species") << ' '
        << pkg->Param<Real>("alpha_weight") << ' '
        << pkg->Param<Real>("cumulative_alphas_sourced") << '\n'
        << std::quoted(PlasmaSignature(pm)) << '\n';
    out << state->roulette_count_delta << ' ' << state->roulette_energy_delta;
    for(auto p:state->roulette_momentum_delta) out << ' ' << p;
    out << '\n';
    if(state->static_mesh) {
      out << state->leaf_geometry.size() << '\n';
      for(auto x:state->leaf_geometry) out << x << ' ';
      out << '\n';
    }
  }
  out << state->deposited << ' ' << state->escaped << ' '
      << state->escaped_count << ' ' << state->thermalized << '\n';
  for (const auto &a : state->particles) {
    for (auto x : a.x)
      out << x << ' ';
    for (auto v : a.v)
      out << v << ' ';
    out << a.time << ' ' << a.weight << '\n';
  }
  out.flush();
  PARTHENON_REQUIRE(out.good(), "Could not write RZ CPT checkpoint");
  out.close();
  std::filesystem::rename(temporary, target);
  // All companions must exist before the hydro checkpoint can be written.
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
}
} // namespace

void Initialize(ParameterInput *pin, StateDescriptor *pkg) {
  const bool plasma=pkg->Param<bool>("tn_plasma");
  PARTHENON_REQUIRE(pin->GetInteger("parthenon/mesh","nx2")>1 &&
      pin->GetInteger("parthenon/mesh","nx3")==1 && pin->GetReal("parthenon/mesh","x1min")==0 &&
      pin->GetReal("parthenon/mesh","x2min")<0 &&
      pin->GetReal("parthenon/mesh","x2max")==-pin->GetReal("parthenon/mesh","x2min"),
      "RZ CPT requires R>=0 and a full symmetric axial domain");
  PARTHENON_REQUIRE(pin->GetReal("parthenon/mesh","x3min")==0 &&
      std::abs(pin->GetReal("parthenon/mesh","x3max")-2*M_PI)<1.e-12,
      "RZ CPT requires a full 2*pi azimuthal extent");
  const auto refinement=pin->GetString("parthenon/mesh","refinement");
  PARTHENON_REQUIRE((refinement=="none" || (plasma && refinement=="static")) &&
      !pkg->Param<bool>("tally_mass_momentum"),
      "RZ plasma supports uniform or static meshes; dynamic AMR is unsupported");
  if (!plasma) {
  PARTHENON_REQUIRE(pin->GetString("parthenon/mesh","refinement")=="none" &&
      !pkg->Param<bool>("tn_plasma") && !pin->GetOrAddBoolean("physics","ionization",false) &&
      !pin->GetOrAddBoolean("physics","tn",false) &&
      !pin->GetOrAddBoolean("physics","multigroup_diffusion",false) &&
      !pin->GetOrAddBoolean("physics","radiation_transport",false) &&
      !pkg->Param<bool>("tally_mass_momentum"),
      "RZ CPT supports only uniform, energy-only constant-stopping verification");
  } else {
    PARTHENON_REQUIRE(pin->GetOrAddBoolean("charon","rz_no_swirl",true),
                      "RZ plasma currently requires rz_no_swirl=true");
    PARTHENON_REQUIRE(pin->GetString("parthenon/mesh","ix1_bc")=="reflecting" &&
        pin->GetString("parthenon/mesh","ox1_bc")=="outflow" &&
        pin->GetString("parthenon/mesh","ix2_bc")=="outflow" &&
        pin->GetString("parthenon/mesh","ox2_bc")=="outflow",
        "RZ plasma requires reflecting axis and escaping outer cylinder/endcaps");
  }
  PARTHENON_REQUIRE(
      !pkg->Param<bool>("weak_scaling_diagnostics") &&
          pkg->Param<bool>("transport_alphas"),
      "RZ CPT requires transport_alphas=true and weak_scaling_diagnostics=false");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||                         \
    defined(KOKKOS_ENABLE_SYCL)
  PARTHENON_FAIL("RZ CPT currently supports CPU builds only");
#endif
  if (!pkg->Param<bool>("tn_plasma")) {
    const auto strategy = pkg->Param<charon::SourceStrategy>("source_strategy");
    PARTHENON_REQUIRE(pkg->Param<std::string>("stopping_power_model") == "constant" &&
                          pin->GetOrAddBoolean("physics", "fixed_fluid", false) &&
                          !pkg->Param<bool>("do_population_control") &&
                          (strategy == charon::SourceStrategy::point ||
                           strategy == charon::SourceStrategy::isotropic),
                      "RZ verification requires constant stopping, fixed_fluid, "
                      "no population control, and point or isotropic sourcing");
    PARTHENON_REQUIRE(pkg->Param<int>("num_particles") >= 0 &&
                          pkg->Param<int>("num_particles_per_pulse") >= 0 &&
                          std::isfinite(pkg->Param<Real>("tlim")) &&
                          pkg->Param<Real>("tlim") > 0,
                      "RZ verification requires nonnegative counts and finite tlim > 0");
    PARTHENON_REQUIRE(std::isfinite(pkg->Param<Real>("stopping_power_value")) &&
                          pkg->Param<Real>("stopping_power_value") >= 0,
                      "RZ verification requires nonnegative finite stopping power");
    const auto outer = pin->GetString("parthenon/mesh", "ox1_bc");
    PARTHENON_REQUIRE(outer == "reflecting" || outer == "outflow",
                      "RZ verification supports reflecting or outflow outer boundaries");
    PARTHENON_REQUIRE(pin->GetString("parthenon/mesh","ix1_bc")=="reflecting" &&
        pin->GetString("parthenon/mesh","ix2_bc")==outer &&
        pin->GetString("parthenon/mesh","ox2_bc")==outer,
        "RZ verification requires matching outer BCs and a reflecting axis");
    pkg->AddParam("verification_reflect_outer", outer == "reflecting");
  }
  PARTHENON_REQUIRE(pkg->Param<int>("source_time_pulses")==1 &&
      pkg->Param<Real>("alpha_source_physical_cap")==std::numeric_limits<Real>::max(),
      "RZ verification requires timestep-start births and no source cap");
  auto state = std::make_shared<State>();
  state->static_mesh=refinement=="static";
  state->nr=pin->GetInteger("parthenon/mesh","nx1");
  state->nz=pin->GetInteger("parthenon/mesh","nx2");
  state->rmax=pin->GetReal("parthenon/mesh","x1max");
  state->zmin=pin->GetReal("parthenon/mesh","x2min");
  state->zmax=pin->GetReal("parthenon/mesh","x2max");
  PARTHENON_REQUIRE(state->nr>0 && state->nz>0 && state->rmax>0 && std::isfinite(state->rmax) &&
                    std::isfinite(state->zmax),"Invalid RZ extent");
  // Independent streams per owner; restart restores the exact saved engine.
  std::seed_seq seed{static_cast<unsigned>(pkg->Param<int>("seed")),
                     static_cast<unsigned>(Globals::my_rank)};
  state->rng.seed(seed);
  state->checkpoint_directory =
      pin->GetOrAddString("charon", "rz_checkpoint_directory",
                          std::filesystem::current_path().string());
  pkg->AddParam("rz_state", state);
  const auto before = pkg->UserWorkBeforeLoopMesh;
  pkg->UserWorkBeforeLoopMesh = [before](Mesh *pm, ParameterInput *input, SimTime &tm) {
    if (before) before(pm, input, tm);
    if (Globals::my_rank == 0 && tm.ncycle == 0 && tm.time == 0) {
      std::ofstream balance("rz_balance.log", std::ios::trunc);
      PARTHENON_REQUIRE(balance.good(), "Could not initialize RZ balance log");
    }
    auto state = GetState(pm);
    auto bg = Gather(pm, false); // Establish immutable leaf geometry before restart checks.
    if (tm.ncycle != 0 || tm.time != 0) {
      std::ifstream in(CheckpointPath(*state, tm.ncycle));
      std::string magic;
      int ranks = -1, rank = -1, cycle = -1;
      Real time = -1;
      std::size_t count = 0;
      in >> magic >> ranks >> rank >> cycle >> time >> state->source_sequence >> count;
      PARTHENON_REQUIRE(in.good() && (pm->packages.Get("charon")->Param<bool>("tn_plasma")
                            ? (state->static_mesh ? magic=="RIOT_RZ_PLASMA_V4"
                               : (magic=="RIOT_RZ_PLASMA_V2" || magic=="RIOT_RZ_PLASMA_V3"))
                            : magic=="RIOT_RZ_CPT_V1") &&
                            ranks == Globals::nranks && rank == Globals::my_rank &&
                            cycle == tm.ncycle && time == tm.time,
                        "Missing/incompatible RZ CPT companion checkpoint; "
                        "restore its directory and use the original MPI rank count");
      in >> state->rng;
      if(pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
        int nr,nz,ns; Real r,zlo,zhi,weight,cumulative;
        in >> nr >> nz >> r >> zlo >> zhi >> ns >> weight >> cumulative;
        auto pkg=pm->packages.Get("charon");
        PARTHENON_REQUIRE(in.good() && nr==state->nr && nz==state->nz && r==state->rmax &&
            zlo==state->zmin && zhi==state->zmax && ns==pkg->Param<int>("plasma_num_species") &&
            weight==pkg->Param<Real>("alpha_weight") && std::isfinite(cumulative) && cumulative>=0,
            "Incompatible RZ plasma geometry/species/weight checkpoint");
        std::string signature; in >> std::quoted(signature);
        PARTHENON_REQUIRE(magic!="RIOT_RZ_PLASMA_V2" || !pkg->Param<bool>("do_population_control"),
                          "Legacy RZ plasma checkpoint requires population control off");
        PARTHENON_REQUIRE(in.good() && signature==PlasmaSignature(pm, (magic=="RIOT_RZ_PLASMA_V3" || magic=="RIOT_RZ_PLASMA_V4")),
                          "Incompatible RZ plasma stopping/species/depletion/population settings");
        if((magic=="RIOT_RZ_PLASMA_V3" || magic=="RIOT_RZ_PLASMA_V4")) {
          in >> state->roulette_count_delta >> state->roulette_energy_delta;
          for(auto &p:state->roulette_momentum_delta) in >> p;
          PARTHENON_REQUIRE(in.good() && std::isfinite(state->roulette_count_delta) &&
              std::isfinite(state->roulette_energy_delta) && std::isfinite(Radius(state->roulette_momentum_delta)),
              "Invalid RZ checkpoint roulette accounting");
        }
        if(magic=="RIOT_RZ_PLASMA_V4") {
          std::size_t size=0; in >> size;
          PARTHENON_REQUIRE(in.good() && size==state->leaf_geometry.size(),"Incompatible SMR leaf count");
          for(auto expected:state->leaf_geometry) {
            Real saved=0; in >> saved;
            PARTHENON_REQUIRE(in.good() && saved==expected,"Incompatible SMR leaf bounds");
          }
        }
        pkg->UpdateParam<Real>("cumulative_alphas_sourced",cumulative);
      }

      in >> state->deposited >> state->escaped >> state->escaped_count >> state->thermalized;
      PARTHENON_REQUIRE(in.good() && std::isfinite(state->deposited) && state->deposited>=0 &&
          std::isfinite(state->escaped) && state->escaped>=0 &&
          std::isfinite(state->escaped_count) && state->escaped_count>=0 &&
          std::isfinite(state->thermalized) && state->thermalized>=0,
          "Invalid RZ checkpoint accounting");
      state->particles.resize(count);
      for (auto &a : state->particles) {
        for (auto &x : a.x)
          in >> x;
        for (auto &v : a.v)
          in >> v;
        in >> a.time >> a.weight;
        PARTHENON_REQUIRE(in.good() && a.time == tm.time && a.weight > 0 && std::isfinite(a.weight) &&
                              std::isfinite(Radius(a.x)) && std::isfinite(Radius(a.v)),
                          "Invalid RZ CPT companion particle state");
      }
      PARTHENON_REQUIRE(in.good(), "Truncated RZ CPT RNG/particle checkpoint");
    }
    if(!pm->packages.Get("charon")->Param<bool>("tn_plasma"))
    pm->packages.Get("charon")->UpdateParam<Real>("cumulative_alphas_sourced",
        state->source_sequence*pm->packages.Get("charon")->Param<Real>("alpha_weight"));
    // Load from the saved directory; future checkpoints belong to this run.
    state->checkpoint_directory = std::filesystem::current_path().string();
    input->SetString("charon", "rz_checkpoint_directory",
                     state->checkpoint_directory);
    // Refresh transient census and feedback workspace after fresh start/restart.

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

void Step(Mesh *pm,Real start,Real dt,int cycle) {
  // Roulette only the incoming census; newborns first become eligible next step.
  RouletteActiveCensus(pm, cycle);
  auto bg=Gather(pm);
  SourceParticles(pm,bg,start,dt);
  std::vector<Real> tallies(bg.Size()*NTally,0);
  int expanded_log_used=0;
  Transport(pm,bg,start+dt,tallies,expanded_log_used);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,&expanded_log_used,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD));
#endif
  auto warning_state=GetState(pm);
  if(expanded_log_used && warning_state->last_lp_warning_cycle!=cycle) {
    if(Globals::my_rank==0) std::cout << "WARNING: Li-Petrasso encountered a negative Coulomb logarithm; using the expanded Coulomb logarithm. The model may be insufficient in this regime. cycle=" << cycle << " time_s=" << start << std::endl;
    warning_state->last_lp_warning_cycle=cycle;
  }
  ActiveTallies(pm,bg,tallies);
  if(pm->packages.Get("charon")->Param<bool>("tn_plasma")) {
    ScatterSelected(pm,bg,tallies);
    return;
  }
  auto state=GetState(pm); Real active=0;
  for (const auto &a:state->particles)
    active+=a.weight*0.5*charon::units::MALPHA*Dot(a.v,a.v);
  std::vector<Real> totals{active,state->deposited,state->escaped,state->escaped_count,state->thermalized};
  Sum(totals);
  const Real injected=state->source_sequence*pm->packages.Get("charon")->Param<Real>("alpha_weight")*charon::units::E0;
  if (Globals::my_rank==0) {
    // Keep the console identical to ordinary Charon sourcing; retain the
    // verification balance separately for the analysis script.
    std::ofstream balance("rz_balance.log", std::ios::app);
    PARTHENON_REQUIRE(balance.good(), "Could not open RZ balance log");
    balance<<std::setprecision(17)<<"[RZ CPT balance] time_s="<<start+dt
      <<" injected_erg="<<injected<<" active_erg="<<totals[0]<<" deposited_erg="<<totals[1]
      <<" escaped_erg="<<totals[2]<<" residual_erg="<<injected-totals[0]-totals[1]-totals[2]
      <<" escaped_alphas="<<totals[3]<<" thermalized_alphas="<<totals[4]<<std::endl;
    PARTHENON_REQUIRE(balance.good(), "Could not write RZ balance log");
  }
  ScatterSelected(pm,bg,tallies);
}
} // namespace CPT::rz
