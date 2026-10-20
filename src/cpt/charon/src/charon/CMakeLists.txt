# ========================================================================================
# (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

# Generate configuration file for charon library to interface with host
configure_file("charon_config.hpp.in"
               ${CMAKE_CURRENT_BINARY_DIR}/charon_config.hpp @ONLY)

# Charon physics package
file(GLOB package_sources "*.cpp")
file(GLOB package_headers "*.hpp")
file(GLOB package_utils "*.hpp")
add_library(lib_charon_package ${package_sources} ${package_headers}
                                 ${package_utils})
target_compile_definitions(lib_charon_package
                           PRIVATE PORTABILITY_STRATEGY_KOKKOS)
target_link_libraries(lib_charon_package PUBLIC
  $<IF:$<BOOL:${RIOT_BUILD_PARTHENON}>,parthenon,Parthenon::parthenon>)
target_include_directories(
  lib_charon_package
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         "${CMAKE_CURRENT_BINARY_DIR}" ${CHARON_SINGULARITY_INCLUDE_PATHS}
)
install(
  TARGETS lib_charon_package
  EXPORT charon-targets
  DESTINATION lib)
install(FILES ${package_headers} DESTINATION include/charon/package)

