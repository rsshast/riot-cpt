if(NOT TARGET Catch2::Catch2)
  add_subdirectory(${CMAKE_SOURCE_DIR}/external/parthenon/external/Catch2
                   ${CMAKE_BINARY_DIR}/catch2)
endif()

add_executable(unit_cpt_tests catch2_main.cpp cpt_transport_tests.cpp)

target_link_libraries(unit_cpt_tests PRIVATE lib_charon_package Catch2::Catch2
                                             Kokkos::kokkos)

target_include_directories(
  unit_cpt_tests
  PRIVATE ${CMAKE_SOURCE_DIR}/src/charon
          ${CMAKE_BINARY_DIR}/src/charon)

add_test(NAME unit_cpt COMMAND unit_cpt_tests)
