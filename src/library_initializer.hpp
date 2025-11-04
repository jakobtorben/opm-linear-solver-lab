#ifndef LIBRARY_INITIALIZER_HPP
#define LIBRARY_INITIALIZER_HPP

#include "config.hpp"

#include <iostream>

#if HAVE_MPI
#include <mpi.h>
#endif

#if HAVE_CUDA
#include <opm/simulators/linalg/gpuistl/set_device.hpp>
#endif

#if HAVE_HYPRE
#include <HYPRE.h>
#include <_hypre_utilities.h>
#endif

#if HAVE_AMGX
#include <amgx_c.h>
#endif

/**
 * @brief RAII wrapper for library initialization and cleanup
 *
 * Automatically initializes MPI, CUDA, HYPRE, and AMGX on construction
 * and cleans them up on destruction, guaranteeing proper cleanup even
 * on exceptions.
 */
class LibraryInitializer
{
public:
    LibraryInitializer(int argc, char** argv)
    {
#if HAVE_MPI
        MPI_Init(&argc, &argv);
#endif

#if HAVE_CUDA
        Opm::gpuistl::setDevice(0, 1);
#endif

#if HAVE_HYPRE
#if HYPRE_RELEASE_NUMBER >= 22900
        HYPRE_Initialize();
#else
        HYPRE_Init();
#endif
        // Print version info
        std::cerr << "\nHYPRE " << HYPRE_RELEASE_NUMBER / 10000 << "." << (HYPRE_RELEASE_NUMBER / 100) % 100 << "."
                  << HYPRE_RELEASE_NUMBER % 100;
#if HYPRE_USING_CUDA
        std::cerr << " (CUDA)";
#elif HYPRE_USING_HIP
        std::cerr << " (HIP)";
#endif
        std::cerr << "\n";
#endif

#if HAVE_AMGX
        AMGX_SAFE_CALL(AMGX_initialize());
#endif
    }

    ~LibraryInitializer()
    {
#if HAVE_HYPRE
#if HYPRE_RELEASE_NUMBER >= 22900
        if (HYPRE_Initialized()) {
            HYPRE_Finalize();
        }
#else
        HYPRE_Finalize();
#endif
#endif

#if HAVE_AMGX
        AMGX_SAFE_CALL(AMGX_finalize());
#endif

#if HAVE_MPI
        MPI_Finalize();
#endif
    }

    // Prevent copying
    LibraryInitializer(const LibraryInitializer&) = delete;
    LibraryInitializer& operator=(const LibraryInitializer&) = delete;
};

#endif // LIBRARY_INITIALIZER_HPP
