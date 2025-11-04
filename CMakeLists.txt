cmake_minimum_required(VERSION 3.27)
# We need this for Damaris
cmake_policy(SET CMP0148 OLD)
project(
  opm-linear-solver-lab
  VERSION 1.0
  LANGUAGES C CXX CUDA)
option(OPM_REQUIRES_DAMARIS OFF)
option(OPM_REQUIRES_VEXCL OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(MPI)
find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)
find_package(dune-common REQUIRED)
find_package(dune-grid REQUIRED)
find_package(dune-geometry REQUIRED)
find_package(dune-istl REQUIRED)

find_package(OpenMP REQUIRED)
find_package(VexCL)

if(VexCL_FOUND)
  get_property(VEXCL_INCLUDE_DIRS TARGET VexCL::Common PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  get_property(VEXCL_LINK_LIBRARIES TARGET VexCL::Common PROPERTY INTERFACE_LINK_LIBRARIES)
  get_property(VEXCL_COMPILE_DEFINITIONS TARGET VexCL::OpenCL PROPERTY INTERFACE_COMPILE_DEFINITIONS)
  set(VEXCL_LINK_LIBRARIES "${VEXCL_LINK_LIBRARIES};OpenCL::OpenCL")
  add_library(OPM::VexCL::OpenCL INTERFACE IMPORTED)
  set_target_properties(OPM::VexCL::OpenCL PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${VEXCL_COMPILE_DEFINITIONS}"
    INTERFACE_LINK_LIBRARIES "${VEXCL_LINK_LIBRARIES}")
  target_include_directories(OPM::VexCL::OpenCL SYSTEM INTERFACE "${VEXCL_INCLUDE_DIRS}")
endif()

find_package(HYPRE)
find_package(AMGX)

find_package(opm-simulators REQUIRED)
find_package(fmt REQUIRED)
find_package(Petsc)
find_package(amgcl)

# HYPRE requires MPI
if(HYPRE_FOUND AND NOT MPI_FOUND)
  message(FATAL_ERROR "HYPRE requires MPI. Please install MPI or disable HYPRE.")
endif()

set(HAVE_AMGX ${AMGX_FOUND})
set(HAVE_HYPRE ${HYPRE_FOUND})
set(HAVE_AMGCL ${amgcl_FOUND})
set(HAVE_CUDA ${CUDAToolkit_FOUND})
set(HAVE_MPI ${MPI_FOUND})

if (OPM_REQUIRES_DAMARIS)
  find_package(XercesC REQUIRED)
  find_package(Damaris REQUIRED)
endif()

if (OPM_REQUIRES_VEXCL)
  include(VexCLSetup)
endif()

find_package(Boost REQUIRED COMPONENTS program_options thread log log_setup python numpy unit_test_framework)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.hpp ESCAPE_QUOTES)

add_executable(linsolverlab src/main.cpp
    src/read_binary.hpp)

target_compile_features(linsolverlab PRIVATE cxx_std_20)
target_include_directories(linsolverlab PRIVATE 
  # Does not seem like these are included in any target? Doing it the old fashioned way...
  ${opm-simulators_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(linsolverlab PRIVATE opmsimulators Boost::program_options CUDA::cudart fmt::fmt)

if(HAVE_MPI)
  target_include_directories(linsolverlab PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(linsolverlab PRIVATE MPI::MPI_CXX)
endif()

if (HAVE_AMGCL)
    target_link_libraries(linsolverlab PRIVATE opmsimulators amgcl::amgcl)

    add_executable(test_read_binary tests/test_read_binary.cpp)
    target_compile_features(test_read_binary PRIVATE cxx_std_20)

    target_include_directories(test_read_binary PRIVATE
      # Does not seem like these are included in any target? Doing it the old fashioned way...
      ${opm-simulators_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR}/include
    )

    enable_testing()
    target_link_libraries(test_read_binary PRIVATE opmsimulators Boost::program_options Boost::unit_test_framework amgcl::amgcl fmt::fmt)

    add_test(NAME read_binary+spe1 COMMAND test_read_binary
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/matrices/spe1)
endif()

if(Petsc_FOUND)
  add_executable(linearsolverlab_petsc src/petsc.cpp)
  target_include_directories(linearsolverlab_petsc PRIVATE
                             ${PETSC_INCLUDE_DIRS}
                             ${opm-simulators_INCLUDE_DIRS}
                             ${CMAKE_CURRENT_BINARY_DIR}/include)
  target_link_libraries(linearsolverlab_petsc PRIVATE ${PETSC_LIBRARIES} fmt::fmt Boost::program_options ${opm-simulators_LIBRARIES})
    target_compile_features(linearsolverlab_petsc PRIVATE cxx_std_20)
endif()

