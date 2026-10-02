// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_KERNEL_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_KERNEL_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_input.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle::Kernel
{
  //! get kernel data
  inline Particle::KernelData parse_kernel_params(const Teuchos::ParameterList& params)
  {
    return {Teuchos::getIntegralValue<Particle::KernelType>(params, "KERNEL"),
        Teuchos::getIntegralValue<Particle::KernelSpaceDimension>(params, "KERNEL_SPACE_DIM")};
  }

  //! get spatial dimension of the kernel
  inline int kernel_space_dimension(const Particle::KernelData data)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (spacedim)
    {
      case Particle::KernelSpaceDimension::Kernel1D:
        return 1;

      case Particle::KernelSpaceDimension::Kernel2D:
        return 2;

      case Particle::KernelSpaceDimension::Kernel3D:
        return 3;

      default:
        FOUR_C_THROW("unknown kernel space dimension!");
    }

    std23::unreachable();
  }

  //! get smoothing length from kernel support radius
  inline double smoothing_length(const Particle::KernelData data, const double& support)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (type)
    {
      case ParticleKernelType::CubicSpline:
        return (0.5 * support);

      case Particle::KernelType::QuinticSpline:
        // (support / 3.0)
        return 0.3333333333333333 * support;
    }
  }

  //! get normalization constant from inverse smoothing length
  inline double normalization_constant(const Particle::KernelData data, const double& inv_h)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (type)
    {
      case Particle::KernelType::CubicSpline:
        switch (spacedim)
        {
          case Particle::KernelSpaceDimension::Kernel1D:
          {
            // (2.0 / 3.0) * inv_h
            return 0.6666666666666666 * inv_h;
          }
          case Particle::KernelSpaceDimension::Kernel2D:
          {
            // (10.0 / 7.0) * std::numbers::inv_pi * inv_h * inv_h
            return 0.4547284088339866 * ParticleUtils::pow<2>(inv_h);
          }
          case Particle::KernelSpaceDimension::Kernel3D:
          {
            return std::numbers::inv_pi * ParticleUtils::pow<3>(inv_h);
          }
          default:
          {
            FOUR_C_THROW("unknown kernel space dimension!");
            break;
          }
        }
      case Particle::KernelType::QuinticSpline:
        switch (spacedim)
        {
          case Particle::KernelSpaceDimension::Kernel1D:
          {
            // (inv_h / 120.0)
            return 0.0083333333333333 * inv_h;
          }
          case Particle::KernelSpaceDimension::Kernel2D:
          {
            // (7.0 / 478.0) * std::numbers::inv_pi * inv_h * inv_h
            return 0.0046614418478797 * Particle::ParticleUtils::pow<2>(inv_h);
          }
          case Particle::KernelSpaceDimension::Kernel3D:
          {
            // (3.0 / 359.0) * std::numbers::inv_pi * inv_h * inv_h * inv_h
            return 0.0026599711937364 * Particle::ParticleUtils::pow<3>(inv_h);
          }
          default:
          {
            FOUR_C_THROW("unknown kernel space dimension!");
            break;
          }
        }
    }
  }

  //! evaluate kernel (self-interaction)
  inline double w0(const Particle::KernelData data, const double& support)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (type)
    {
      case ParticleKernelType::CubicSpline:
        return normalization_constant(data, 2.0 / support);
      case ParticleKernelType::QuinticSpline:
        return 66.0 * normalization_constant(data, 3.0 / support);
    }
  }

  //! evaluate kernel
  inline double w(const Particle::KernelData data, const double& rij, const double& support)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (type)
    {
      case ParticleKernelType::CubicSpline:
      {
        const double inv_h = 2.0 / support;
        const double q = rij * inv_h;

        if (q < 1.0)
          return (1.0 - 1.5 * Particle::ParticleUtils::pow<2>(q) +
                     0.75 * Particle::ParticleUtils::pow<3>(q)) *
                 normalization_constant(data, inv_h);
        else if (q < 2.0)
          return (0.25 * Particle::ParticleUtils::pow<3>(2.0 - q)) *
                 normalization_constant(data, inv_h);
        else
          return 0.0;
      }
      case ParticleKernelType::QuinticSpline:
      {
        const double inv_h = 3.0 / support;
        const double q = rij * inv_h;

        if (q < 1.0)
          return (Particle::ParticleUtils::pow<5>(3.0 - q) -
                     6.0 * Particle::ParticleUtils::pow<5>(2.0 - q) +
                     15.0 * Particle::ParticleUtils::pow<5>(1.0 - q)) *
                 normalization_constant(data, inv_h);
        else if (q < 2.0)
          return (Particle::ParticleUtils::pow<5>(3.0 - q) -
                     6.0 * Particle::ParticleUtils::pow<5>(2.0 - q)) *
                 normalization_constant(data, inv_h);
        else if (q < 3.0)
          return Particle::ParticleUtils::pow<5>(3.0 - q) * normalization_constant(data, inv_h);
        else
          return 0.0;
      }
    }
  }

  //! evaluate first derivative of kernel
  inline double d_wdrij(const Particle::KernelData data, const double& rij, const double& support)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (type)
    {
      case ParticleKernelType::CubicSpline:
      {
        const double inv_h = 2.0 / support;
        const double q = rij * inv_h;

        if (q < 1.0)
          return (-3.0 * q + 2.25 * Particle::ParticleUtils::pow<2>(q)) * inv_h *
                 normalization_constant(data, inv_h);
        else if (q < 2.0)
          return (-0.75 * Particle::ParticleUtils::pow<2>(2.0 - q)) * inv_h *
                 normalization_constant(data, inv_h);
        else
          return 0.0;
      }
      case ParticleKernelType::QuinticSpline:
      {
        const double inv_h = 3.0 / support;
        const double q = rij * inv_h;

        if (q < 1.0)
          return (-5.0 * Particle::ParticleUtils::pow<4>(3.0 - q) +
                     30.0 * Particle::ParticleUtils::pow<4>(2.0 - q) -
                     75.0 * Particle::ParticleUtils::pow<4>(1.0 - q)) *
                 inv_h * normalization_constant(data, inv_h);
        else if (q < 2.0)
          return (-5.0 * Particle::ParticleUtils::pow<4>(3.0 - q) +
                     30.0 * Particle::ParticleUtils::pow<4>(2.0 - q)) *
                 inv_h * normalization_constant(data, inv_h);
        else if (q < 3.0)
          return (-5.0 * Particle::ParticleUtils::pow<4>(3.0 - q)) * inv_h *
                 normalization_constant(data, inv_h);
        else
          return 0.0;
      }
    }
  }

  inline void grad_wij(const Particle::KernelData data, const double& rij, const double& support,
      const double* eij, double* gradWij)
  {
    Particle::ParticleUtils::vec_set_scale(gradWij, d_wdrij(data, rij, support), eij);
  }

  //! evaluate second derivative of kernel
  inline double d2_wdrij2(const Particle::KernelData data, const double& rij, const double& support)
  {
    Particle::KernelSpaceDimension spacedim;
    Particle::KernelType type;
    std::tie(type, spacedim) = data;

    switch (type)
    {
      case ParticleKernelType::CubicSpline:
      {
        const double inv_h = 2.0 / support;
        const double q = rij * inv_h;

        if (q < 1.0)
          return (-3.0 + 4.5 * q) * Particle::ParticleUtils::pow<2>(inv_h) *
                 normalization_constant(data, inv_h);
        else if (q < 2.0)
          return (1.5 * (2.0 - q)) * Particle::ParticleUtils::pow<2>(inv_h) *
                 normalization_constant(data, inv_h);
        else
          return 0.0;
      }
      case ParticleKernelType::QuinticSpline:
      {
        const double inv_h = 3.0 / support;
        const double q = rij * inv_h;

        if (q < 1.0)
          return (20.0 * Particle::ParticleUtils::pow<3>(3.0 - q) -
                     120.0 * Particle::ParticleUtils::pow<3>(2.0 - q) +
                     300.0 * Particle::ParticleUtils::pow<3>(1.0 - q)) *
                 Particle::ParticleUtils::pow<2>(inv_h) * normalization_constant(data, inv_h);
        else if (q < 2.0)
          return (20.0 * Particle::ParticleUtils::pow<3>(3.0 - q) -
                     120.0 * Particle::ParticleUtils::pow<3>(2.0 - q)) *
                 Particle::ParticleUtils::pow<2>(inv_h) * normalization_constant(data, inv_h);
        else if (q < 3.0)
          return (20.0 * Particle::ParticleUtils::pow<3>(3.0 - q)) *
                 Particle::ParticleUtils::pow<2>(inv_h) * normalization_constant(data, inv_h);
        else
          return 0.0;
      }
    }
  }

}  // namespace Particle::Kernel

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
