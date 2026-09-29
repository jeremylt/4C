// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_surface_tension_recoilpressure_evaporation.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_utils.hpp"

#include <Kokkos_Core.hpp>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHRecoilPressureEvaporation::SPHRecoilPressureEvaporation(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      evaporatingphase_(ParticleType::Phase1),
      recoilboilingtemp_(params_sph_.get<double>("VAPOR_RECOIL_BOILINGTEMPERATURE")),
      recoil_pfac_(params_sph_.get<double>("VAPOR_RECOIL_PFAC")),
      recoil_tfac_(params_sph_.get<double>("VAPOR_RECOIL_TFAC"))
{
  // empty constructor
}

void Particle::SPHRecoilPressureEvaporation::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();
}

void Particle::SPHRecoilPressureEvaporation::compute_recoil_pressure_contribution() const
{
  // get container of owned particles of evaporating phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle_->get_specific_container(evaporatingphase_, ParticleStatus::Owned);

  // get pointers to particle states
  const int statedim = Particle::enum_to_state_dim(ParticleState::Position);
  const double* dens = container_i->get_ptr_to_state(ParticleState::Density, ParticleSpace::Device);
  const double* temp =
      container_i->get_ptr_to_state(ParticleState::Temperature, ParticleSpace::Device);
  const double* cfg =
      container_i->get_ptr_to_state(ParticleState::ColorfieldGradient, ParticleSpace::Device);
  const double* ifn =
      container_i->get_ptr_to_state(ParticleState::InterfaceNormal, ParticleSpace::Device);
  double* acc =
      container_i->get_ptr_to_state_writable(ParticleState::Acceleration, ParticleSpace::Device);

  // iterate over particles in container
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, container_i->particles_stored()),
      KOKKOS_CLASS_LAMBDA(const int particle_i) {
        // get pointers to states
        const double* dens_i = &dens[particle_i];
        const double* temp_i = &temp[particle_i];
        const double* cfg_i = &cfg[particle_i * statedim];
        const double* ifn_i = &ifn[particle_i * statedim];
        double* acc_i = &acc[particle_i * statedim];

        // evaluation only for non-zero interface normal
        if ((ParticleUtils::vec_norm_two(ifn_i) > 0.0) and (temp_i[0] > recoilboilingtemp_))
        {
          // compute evaporation induced recoil pressure
          const double recoil_press_i =
              recoil_pfac_ * std::exp(-recoil_tfac_ * (1.0 / temp_i[0] - 1.0 / recoilboilingtemp_));

          // add contribution to acceleration
          ParticleUtils::vec_add_scale(acc_i, -recoil_press_i / dens_i[0], cfg_i);
        }
      });
}

FOUR_C_NAMESPACE_CLOSE
