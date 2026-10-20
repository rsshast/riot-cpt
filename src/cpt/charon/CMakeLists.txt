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

# This file was made in part with generative AI.

foreach(required_host_variable
    CHARON_HOST_VARIABLE_HEADER
    CHARON_HOST_DENSITY_VARIABLE
    CHARON_HOST_TEMPERATURE_VARIABLE
    CHARON_HOST_ALPHA_SOURCE_VARIABLE)
  if(NOT DEFINED ${required_host_variable})
    message(FATAL_ERROR "RIOT must define ${required_host_variable} for Charon")
  endif()
endforeach()

add_subdirectory(src/charon)
