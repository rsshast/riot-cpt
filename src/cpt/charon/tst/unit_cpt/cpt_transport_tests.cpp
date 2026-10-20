#include <catch2/catch.hpp>

#include <cmath>

#include "charon.hpp"
#include "charon_utils.hpp"
#include "source_profiles.hpp"
#include "stopping_power.hpp"
#include "transport_utils.hpp"
#include "units.hpp"

namespace {

using charon::MeanStoppingPower;
using charon::SineTestParticleCount;
using charon::TemperatureLinearStoppingPower;
using charon::tran_step_args;
using charon::ptcl_transport_step;
using charon::units::EB;
using charon::units::EPS;
using charon::units::MALPHA;
using charon::units::Real;

struct StepFixture {
  RngPool rng_pool;
  RngGen rng_gen;

  Real t_start = 0.0;
  Real dt = 10.0;
  Real ee = 10.0 * EB;
  Real ee_f = ee;
  Real vv_i = 4.0;
  Real vv_f = vv_i;
  Real vx = vv_i;
  Real vy = 0.0;
  Real vz = 0.0;
  Real dx_push = 0.25;
  bool multi_d = false;
  bool three_d = false;
  Real xl = 0.0;
  Real yl = 0.0;
  Real zl = 0.0;
  Real xu = 1.0;
  Real yu = 1.0;
  Real zu = 1.0;
  Real spower = 0.0;
  Real t = 0.0;
  Real x = 0.5;
  Real y = 0.5;
  Real z = 0.5;
  bool in_background = false;

  StepFixture() : rng_pool(12345), rng_gen(rng_pool.get_state()) {}

  ~StepFixture() { rng_pool.free_state(rng_gen); }

  tran_step_args args() {
    return {rng_gen,     t_start, dt,      ee,    ee_f,  vv_i,  vv_f,
            vx,          vy,      vz,      dx_push, multi_d, three_d,
            xl,          yl,      zl,      xu,    yu,    zu,    spower,
            t,           x,       y,       z,     in_background};
  }
};

constexpr Real eps_imc_offset() { return 1.0e6 * EPS; }

} // namespace

TEST_CASE("Sine-test source uses deck amplitudes for initial and later cohorts",
          "[cpt][sourcing]") {
  constexpr Real tf = 1.0e-10;
  constexpr int n0 = 10000;
  constexpr int amplitude = 100;

  REQUIRE(SineTestParticleCount(Real(0), tf, n0, amplitude) == n0);
  REQUIRE(SineTestParticleCount(tf / 4, tf, n0, amplitude) == 70);
  REQUIRE(SineTestParticleCount(tf / 2, tf, n0, amplitude) == amplitude);
  REQUIRE(SineTestParticleCount(tf, tf, n0, amplitude) == 0);
}

TEST_CASE("Mean stopping power removes energy linearly and floors at background",
          "[cpt][stopping_power]") {
  const MeanStoppingPower spower(0.5 * EB);

  REQUIRE(spower.ELoss(10.0 * EB, 2.0) == Approx(9.0 * EB));
  REQUIRE(spower.ELoss(1.5 * EB, 2.0) == Approx(EB));
}

TEST_CASE("Temperature-linear stopping power scales from its reference temperature",
          "[cpt][stopping_power]") {
  const TemperatureLinearStoppingPower spower(2.0, 100.0, 1.0);

  REQUIRE(spower.Value(100.0) == Approx(2.0));
  REQUIRE(spower.Value(150.0) == Approx(3.0));
  REQUIRE(spower.Value(200.0) == Approx(4.0));
}

TEST_CASE("Temperature-linear stopping power clips negative values at zero",
          "[cpt][stopping_power]") {
  const TemperatureLinearStoppingPower spower(2.0, 100.0, 2.0);

  REQUIRE(spower.Value(25.0) == Approx(0.0));
}

TEST_CASE("Temperature-linear stopping power removes energy and floors at background",
          "[cpt][stopping_power]") {
  const TemperatureLinearStoppingPower spower(0.5 * EB, 100.0, 1.0);

  REQUIRE(spower.ELoss(10.0 * EB, 2.0, 200.0) == Approx(8.0 * EB));
  REQUIRE(spower.ELoss(1.5 * EB, 2.0, 200.0) == Approx(EB));
}

TEST_CASE("CPT transport with no stopping power advances linearly", "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 0.25;
  fix.vv_i = 4.0;
  fix.vv_f = 0.0;
  fix.vx = 4.0;
  fix.x = 0.25;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.t == Approx(0.0625));
  REQUIRE(fix.x == Approx(0.5));
  REQUIRE(fix.y == Approx(0.5));
  REQUIRE(fix.z == Approx(0.5));
  REQUIRE(fix.vv_f == Approx(fix.vv_i));
}

TEST_CASE("Constant stopping power computes final speed from work over distance",
          "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 0.5;
  fix.vv_i = 3.0e8;
  fix.vv_f = 0.0;
  fix.vx = fix.vv_i;
  fix.x = -1.0;
  fix.xl = -10.0;
  fix.xu = 10.0;

  const Real expected_vf = 2.0e8;
  fix.spower =
      0.5 * MALPHA * (fix.vv_i * fix.vv_i - expected_vf * expected_vf) / fix.dx_push;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.vv_f == Approx(expected_vf).epsilon(1.0e-12));
}

TEST_CASE("Constant stopping power uses average speed for the push time",
          "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 0.5;
  fix.vv_i = 3.0e8;
  fix.vv_f = 0.0;
  fix.vx = fix.vv_i;
  fix.x = -1.0;
  fix.xl = -10.0;
  fix.xu = 10.0;

  const Real expected_vf = 2.0e8;
  fix.spower =
      0.5 * MALPHA * (fix.vv_i * fix.vv_i - expected_vf * expected_vf) / fix.dx_push;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.t == Approx(fix.dx_push / (0.5 * (fix.vv_i + expected_vf))));
}

TEST_CASE("Constant stopping power can thermalize a particle over a push",
          "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 0.5;
  fix.vv_i = 3.0e8;
  fix.vv_f = -1.0;
  fix.vx = fix.vv_i;
  fix.x = -1.0;
  fix.xl = -10.0;
  fix.xu = 10.0;
  fix.spower = 0.5 * MALPHA * fix.vv_i * fix.vv_i / fix.dx_push;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.vv_f == Approx(0.0));
  REQUIRE(fix.t == Approx(fix.dx_push / (0.5 * fix.vv_i)));
}

TEST_CASE("CPT transport ignores disabled transverse dimensions", "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 1.0;
  fix.vv_i = 5.0;
  fix.vv_f = 0.0;
  fix.vx = 3.0;
  fix.vy = 4.0;
  fix.vz = 12.0;
  fix.x = 2.0;
  fix.y = 7.0;
  fix.z = 9.0;
  fix.xl = -10.0;
  fix.xu = 10.0;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.x == Approx(2.6));
  REQUIRE(fix.y == Approx(7.0));
  REQUIRE(fix.z == Approx(9.0));
}

TEST_CASE("CPT transport advances enabled transverse dimensions", "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 1.0;
  fix.vv_i = 5.0;
  fix.vv_f = 0.0;
  fix.vx = 3.0;
  fix.vy = 4.0;
  fix.vz = 12.0;
  fix.multi_d = true;
  fix.three_d = false;
  fix.x = 2.0;
  fix.y = 7.0;
  fix.z = 9.0;
  fix.xl = -10.0;
  fix.xu = 10.0;
  fix.yl = -10.0;
  fix.yu = 10.0;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.x == Approx(2.6));
  REQUIRE(fix.y == Approx(7.8));
  REQUIRE(fix.z == Approx(9.0));
}

TEST_CASE("CPT transport offsets particles that land on an outgoing face",
          "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 0.25;
  fix.vv_i = 4.0;
  fix.vx = 4.0;
  fix.x = 0.75;
  fix.xl = 0.0;
  fix.xu = 1.0;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.x == Approx(1.0 + eps_imc_offset()));
}

TEST_CASE("CPT transport does not face-offset particles joining the background",
          "[cpt][transport]") {
  StepFixture fix;
  fix.dx_push = 0.25;
  fix.vv_i = 4.0;
  fix.vx = 4.0;
  fix.x = 0.75;
  fix.xl = 0.0;
  fix.xu = 1.0;
  fix.in_background = true;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.x == Approx(1.0));
}

TEST_CASE("CPT transport nudges census particles beyond the time boundary",
          "[cpt][transport]") {
  StepFixture fix;
  fix.dt = 0.5;
  fix.dx_push = 1.0;
  fix.vv_i = 2.0;
  fix.vx = 2.0;
  fix.x = 0.0;
  fix.xl = -10.0;
  fix.xu = 10.0;

  ptcl_transport_step(fix.args());

  REQUIRE(fix.t > fix.dt);
  REQUIRE(fix.t == Approx((1.0 + eps_imc_offset()) * fix.dt));
}
