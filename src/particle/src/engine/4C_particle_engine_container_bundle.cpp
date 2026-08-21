// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_container_bundle.hpp"

#include "4C_particle_engine_object.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
struct Particle::ParticleContainerBundle::TypeStatusStateImpl
{
  // particle state arrays in Kokkos Views indexed by particle state enum
  std::vector<Kokkos::View<TypeStatusStatePointerBundle>> host_;
  std::vector<Kokkos::DualView<TypeStatusStatePointerBundle>> dual_;
  std::vector<bool> is_dual_valid_;
};

Particle::ParticleContainerBundle::ParticleContainerBundle()
{
  // empty constructor
}

Particle::ParticleContainerBundle::~ParticleContainerBundle() = default;

void Particle::ParticleContainerBundle::setup(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes)
{
  std::shared_ptr<ParticleContainer> container;

  // determine necessary size of vector for particle types
  const int typevectorsize = static_cast<int>((--particlestatestotypes.end())->first) + 1;

  // allocate memory to hold particle types
  containers_.resize(typevectorsize);

  // tracking number of states
  int typestatusstatevectorsize_ = 0;

  // iterate over particle types
  for (const auto& typeIt : particlestatestotypes)
  {
    // get particle type
    ParticleType type = typeIt.first;

    // insert particle type into set of stored containers
    storedtypes_.insert(type);

    // allocate memory for container of owned and ghosted particles
    (containers_[static_cast<int>(type)]).resize(2);

    // set of particle state enums of current particle type (equal for owned and ghosted particles)
    const std::set<ParticleState>& stateset = typeIt.second;

    // track max number of states
    typestatusstatevectorsize_ =
        max(typestatusstatevectorsize_, static_cast<int>(--stateset.end()) + 1);

    // initial size of particle container
    int initialsize = 1;

    // create container of owned particles
    container = std::make_shared<ParticleContainer>();
    container->setup(initialsize, stateset);
    // set container of owned particles
    (containers_[static_cast<int>(type)])[static_cast<int>(ParticleStatus::Owned)] = container;

    // create container of ghosted particles
    container = std::make_shared<ParticleContainer>();
    // setup container of ghosted particles
    container->setup(initialsize, stateset);
    // set container of ghosted particles
    (containers_[static_cast<int>(type)])[static_cast<int>(ParticleStatus::Ghosted)] = container;
  }

  // update TypeStatusState vectors
  typestatusstate_->host_.resize(typestatusstatevectorsize_);
  typestatusstate_->dual_.resize(typestatusstatevectorsize_);
  typestatusstate_->is_dual_valid_.resize(typestatusstatevectorsize_);
}

void Particle::ParticleContainerBundle::get_packed_particle_objects_of_all_containers(
    std::vector<char>& particlebuffer) const
{
  // iterate over particle types
  for (const auto& type : storedtypes_)
  {
    // get container of owned particles
    ParticleContainer* container =
        (containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)].get();

    // loop over particles in container
    for (int index = 0; index < container->particles_stored(); ++index)
    {
      int globalid(0);
      ParticleStates states;
      container->get_particle(index, globalid, states);

      ParticleObject particleobject(type, globalid, states);

      // pack data for writing
      Core::Communication::PackBuffer data;
      particleobject.pack(data);
      particlebuffer.insert(particlebuffer.end(), data().begin(), data().end());
    }
  }
}

void Particle::ParticleContainerBundle::get_vector_of_particle_objects_of_all_containers(
    std::vector<ParticleObjShrdPtr>& particlesstored) const
{
  // iterate over particle types
  for (const auto& type : storedtypes_)
  {
    // get container of owned particles
    ParticleContainer* container =
        (containers_[static_cast<int>(type)])[static_cast<int>(ParticleStatus::Owned)].get();

    // loop over particles in container
    for (int index = 0; index < container->particles_stored(); ++index)
    {
      int globalid(0);
      ParticleStates states;
      container->get_particle(index, globalid, states);

      particlesstored.emplace_back(std::make_shared<ParticleObject>(type, globalid, states));
    }
  }
}

inline void Particle::ParticleContainerBundle::get_ptr_to_state_all_containers_internal(
    TypeStatusStatePointerBundle* stateptrs, ParticleState state, ParticleSpace space) const
{
  const int state_idx = static_cast<int>(state);
  if (space == ParticleSpace::Device)
  {
    if (!typestatusstate_->is_dual_valid_[state_idx])
      init_state_dual(state);
    else
      typestatusstate_->dual_[state_idx].sync<Kokkos::DefaultExecutionSpace>();
    stateptrs = typestatusstate_->dual_[state_idx].view<Kokkos::DefaultExecutionSpace>().data();
  }
  else
  {
    stateptrs = typestatusstate_->host_[state_idx].data();
  }
}

void Particle::ParticleContainerBundle::get_ptr_to_state_all_containers(
    ConstTypeStatusStatePointerBundle* stateptrs, ParticleState state, ParticleSpace space) const
{
  const int state_idx = static_cast<int>(state);

  for (auto type : storedtypes_)
  {
    const int type_idx = static_cast<int>(type);

    typestatusstate_->host_[state_idx]
        .data()[0][type_idx][static_cast<int>(ParticleStatus::Owned)] =
        const_cast<double*>(&((containers_[type_idx])[static_cast<int>(ParticleStatus::Owned)])
                ->get_ptr_to_state(state, 0, space));
    typestatusstate_->host_[state_idx]
        .data()[0][type_idx][static_cast<int>(ParticleStatus::Ghosted)] =
        const_cast<double*>(&((containers_[type_idx])[static_cast<int>(ParticleStatus::Ghosted)])
                ->get_ptr_to_state(state, 0, space));
  }
  get_ptr_to_state_all_containers_internal(stateptrs, state, space);
}

void Particle::ParticleContainerBundle::try_get_ptr_to_state_all_containers(
    ConstTypeStatusStatePointerBundle* stateptrs, ParticleState state, ParticleSpace space) const
{
  const int state_idx = static_cast<int>(state);

  for (auto type : storedtypes_)
  {
    const int type_idx = static_cast<int>(type);

    typestatusstate_->host_[state_idx].data()[type_idx][static_cast<int>(ParticleStatus::Owned)] =
        const_cast<double*>(&((containers_[type_idx])[static_cast<int>(ParticleStatus::Owned)])
                ->try_get_ptr_to_state(state, 0, space));
    typestatusstate_->host_[state_idx].data()[type_idx][static_cast<int>(ParticleStatus::Ghosted)] =
        const_cast<double*>(&((containers_[type_idx])[static_cast<int>(ParticleStatus::Ghosted)])
                ->try_get_ptr_to_state(state, 0, space));
  }
  get_ptr_to_state_all_containers_internal(
      static_cast<TypeStatusStatePointerBundle*>(stateptrs), state, space);
}

void Particle::ParticleContainerBundle::get_ptr_to_state_writable_all_containers(
    TypeStatusStatePointerBundle* stateptrs, ParticleState state, ParticleSpace space)
{
  const int state_idx = static_cast<int>(state);

  for (auto type : storedtypes_)
  {
    const int type_idx = static_cast<int>(type);

    typestatusstate_->host_[state_idx].data()[type_idx][static_cast<int>(ParticleStatus::Owned)] =
        ((containers_[type_idx])[static_cast<int>(ParticleStatus::Owned)])
            ->get_ptr_to_state_writable(state, 0, space);
    typestatusstate_->host_[state_idx].data()[type_idx][static_cast<int>(ParticleStatus::Ghosted)] =
        ((containers_[type_idx])[static_cast<int>(ParticleStatus::Ghosted)])
            ->get_ptr_to_state_writable(state, 0, space);
  }
  get_ptr_to_state_all_containers_internal(
      static_cast<TypeStatusStatePointerBundle*>(stateptrs), state, space);
}

void Particle::ParticleContainerBundle::try_get_ptr_to_state_writable_all_container(
    TypeStatusStatePointerBundle* stateptrs, ParticleState state, ParticleSpace space)
{
  const int state_idx = static_cast<int>(state);

  for (auto type : storedtypes_)
  {
    const int type_idx = static_cast<int>(type);

    typestatusstate_->host_[state_idx].data()[type_idx][static_cast<int>(ParticleStatus::Owned)] =
        ((containers_[type_idx])[static_cast<int>(ParticleStatus::Owned)])
            ->try_get_ptr_to_state_writable(state, 0, space);
    typestatusstate_->host_[state_idx].data()[type_idx][static_cast<int>(ParticleStatus::Ghosted)] =
        ((containers_[type_idx])[static_cast<int>(ParticleStatus::Ghosted)])
            ->try_get_ptr_to_state_writable(state, 0, space);
  }
  get_ptr_to_state_all_containers_internal(stateptrs, state, space);
}


void Particle::ParticleContainerBundle::init_state_dual(ParticleState state) const
{
  const int state_idx = static_cast<int>(state);

  if (typestatusstate_->is_dual_valid_[state_idx]) return;
  Kokkos::View<double*> device_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space(), typestatusstate_->host_[state_idx]);
  typestatusstate_->dual_[state_idx] =
      Kokkos::DualView<double*>(device_view, typestatusstate_->host_[state_idx]);
  typestatusstate_->is_dual_valid_[state_idx] = true;
}

FOUR_C_NAMESPACE_CLOSE
