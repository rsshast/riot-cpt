# ========================================================================================
# (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights
# reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for
# Los Alamos National Laboratory (LANL), which is operated by Triad National
# Security, LLC for the U.S. Department of Energy/National Nuclear Security
# Administration. All rights in the program are reserved by Triad National
# Security, LLC, and the U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others acting on its
# behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare derivative works, distribute copies to the public,
# perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Set configuration values for charon library to interface with host
# NOTE(@pdmullen): Default values provided here are mcblock-specific
if(CHARON_STANDALONE_MODE)

  set(CHARON_HOST_VARIABLE_HEADER
      "../mcblock/mcblock_variables.hpp"
      CACHE STRING "Path to host code variables header file" FORCE)
  set(CHARON_HOST_DENSITY_VARIABLE
      "field::material::density"
      CACHE STRING "Variable type name for host density" FORCE)
  set(CHARON_HOST_SPECIFIC_INTERNAL_ENERGY_VARIABLE
      "field::material::sie"
      CACHE STRING "Variable type name for host internal energy density" FORCE)
  set(CHARON_HOST_UPDATE_DENSITY_VARIABLE
      "field::material::density"
      CACHE STRING "Variable type name for host density update" FORCE)
  set(CHARON_HOST_UPDATE_ENERGY_VARIABLE
      "field::material::internal_energy"
      CACHE STRING "Variable type name for host internal energy density" FORCE)
  set(CHARON_HOST_ALPHA_SOURCE_VARIABLE
      "field::material::Q"
      CACHE STRING "Variable type name for host alpha source field" FORCE)
  # Set locations of singularity include files for charon library to use
  # during its configuration step
  set(CHARON_SINGULARITY_INCLUDE_PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils/ports-of-call"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils/spiner"
      "${CMAKE_CURRENT_SOURCE_DIR}/../external/singularity-eos/utils/variant/include"
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

  # singularity eos
  message("Configuring singularity-eos for mcblock")
  add_subdirectory(${CMAKE_SOURCE_DIR}/external/singularity-eos ${CMAKE_BINARY_DIR}/singularity-eos)

endif()

if(NOT DEFINED CHARON_HOST_UPDATE_DENSITY_VARIABLE)
  if(CHARON_HOST_DENSITY_VARIABLE STREQUAL "gas::prim::density")
    set(CHARON_HOST_UPDATE_DENSITY_VARIABLE
        "gas::cons::density"
        CACHE STRING "Variable type name for host density update")
  else()
    set(CHARON_HOST_UPDATE_DENSITY_VARIABLE
        "${CHARON_HOST_DENSITY_VARIABLE}"
        CACHE STRING "Variable type name for host density update")
  endif()
endif()

if(NOT CHARON_STANDALONE_MODE AND NOT DEFINED CHARON_HOST_UPDATE_MOMENTUM_VARIABLE)
  set(CHARON_HOST_UPDATE_MOMENTUM_VARIABLE
      "gas::cons::momentum"
      CACHE STRING "Variable type name for host momentum update")
endif()

if(NOT CHARON_STANDALONE_MODE AND
   NOT DEFINED CHARON_HOST_UPDATE_TOTAL_ENERGY_VARIABLE)
  set(CHARON_HOST_UPDATE_TOTAL_ENERGY_VARIABLE
      "gas::cons::total_energy"
      CACHE STRING "Variable type name for host total energy update")
endif()

if(DEFINED CHARON_HOST_UPDATE_MOMENTUM_VARIABLE)
  set(CHARON_ENABLE_MOMENTUM_COUPLING TRUE)
endif()

if(DEFINED CHARON_HOST_UPDATE_TOTAL_ENERGY_VARIABLE AND
   NOT "${CHARON_HOST_UPDATE_TOTAL_ENERGY_VARIABLE}" STREQUAL
       "${CHARON_HOST_UPDATE_ENERGY_VARIABLE}")
  set(CHARON_ENABLE_TOTAL_ENERGY_COUPLING TRUE)
endif()

# Build charon library
add_subdirectory(charon)

# Build mcblock driver
if(CHARON_STANDALONE_MODE)
  add_subdirectory(mcblock)
endif()
