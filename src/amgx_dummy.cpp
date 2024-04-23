#include <petscksp.h>
#include <petscconf.h>

#ifndef PETSC_HAVE_MPIUNI
#define HAVE_MPI 1
#endif

#include <boost/program_options.hpp>

#include <opm/simulators/utils/DeferredLoggingErrorHelpers.hpp>
#include <opm/simulators/utils/ParallelCommunication.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

Mat readMatrix(const std::string& path)
{
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetFromOptions(A);

    Opm::Parallel::Communication comm(PETSC_COMM_WORLD);

    std::array<int,3> info;
    std::ifstream in;
    std::string line;

    OPM_BEGIN_PARALLEL_TRY_CATCH()
    if (comm.rank() == 0) {
        in.open(path);

        if (!in.good()) {
            throw std::runtime_error("Error loading matrix " + path);
        }

        std::getline(in, line);
        if (line != "%%MatrixMarket matrix coordinate real general") {
            throw std::runtime_error("Unexpected header in matrix " + path + ": " + line);
        }

        std::stringstream str;

        std::getline(in, line);
        str.str(line);
        str >> line >> line >> line >> info[0];

        if (line != "blocked") {
            throw std::runtime_error("Expect a blocked matrix");
        }

        std::getline(in, line);
        str.str(line);
        str >> info[1] >> info[1] >> info[2];
    }
    OPM_END_PARALLEL_TRY_CATCH("readMatrix", comm);

    comm.broadcast(info.data(), 3, 0);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, info[1], info[1]);
    MatMPIAIJSetPreallocation(A, info[2] / info[1], PETSC_NULLPTR, info[2] / info[1], PETSC_NULLPTR);
    MatSetBlockSizes(A, info[0], info[0]);
    MatSetUp(A);
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    if (comm.rank() == 0) {
        std::stringstream str;
        while (std::getline(in, line)) {
            str.clear();
            str.str(line);
            int i, j;
            double val;
            str >> i >> j >> val;
             MatSetValue(A, i-1, j-1, val, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    return A;
}

Vec readVector(const std::string& path)
{
    Vec b;
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetFromOptions(b);

    Opm::Parallel::Communication comm(PETSC_COMM_WORLD);

    std::ifstream in(path);
    std::string line;
    std::array<int,2> info;

    OPM_BEGIN_PARALLEL_TRY_CATCH()
    if (comm.rank() == 0) {
        if (!in.good()) {
            throw std::runtime_error("Error loading vector " + path);
        }

        std::getline(in, line);
        if (line != "%%MatrixMarket matrix array real general") {
            throw std::runtime_error("Unexpected header in vector " + path + ": " + line);
        }

        std::stringstream str;
        int sanity;

        std::getline(in, line);
        str.str(line);
        str >> line >> line >> line >> info[0] >> sanity;

        if (line != "blocked" || sanity != 1) {
            throw std::runtime_error("Expect a blocked vector");
        }

        std::getline(in, line);
        str.clear();
        str.str(line);
        str >> info[1] >> sanity;
        if (sanity != 1) {
            throw std::runtime_error("Expect a vector");
        }
    }
    OPM_END_PARALLEL_TRY_CATCH("readVector", comm);

    comm.broadcast(info.data(), 2, 0);
    VecSetSizes(b, PETSC_DECIDE, info[1]);
    VecSetBlockSize(b, info[0]);
    VecSetFromOptions(b);
    VecSetUp(b);

    if (comm.rank() == 0) {
        int pos = 0;
        std::stringstream str;
        while (std::getline(in, line)) {
            str.clear();
            str.str(line);
            double val;
            str >> val;
            VecSetValue(b, pos++, val, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    return b;
}
static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

int main(int argc, char** argv)
{
    Vec         x, b, u; /* approx solution, RHS, exact solution */
    Mat         A;       /* linear system matrix */
    KSP         ksp;     /* linear solver context */
    PC          pc;      /* preconditioner context */
    PetscReal   norm;    /* norm of solution error */
    PetscInt    i, n = 10, col[3], its;
    PetscMPIInt size;
    PetscScalar value[3];

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Compute the matrix and right-hand-side vector that define
            the linear system, Ax = b.
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
        Create vectors.  Note that we form 1 vector from scratch and
        then duplicate as needed.
    */
    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));

    /*
        Create matrix.  When using MatCreate(), the matrix format can
        be specified at runtime.

        Performance tuning note:  For problems of substantial size,
        preallocation of matrix memory is crucial for attaining good
        performance. See the matrix chapter of the users manual for details.
    */
    PetscCall(MatCreate(PETSC_COMM_SELF, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));

    /*
        Assemble matrix
    */
    value[0] = -1.0;
    value[1] = 2.0;
    value[2] = -1.0;
    for (i = 1; i < n - 1; i++) {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
    }
    i      = n - 1;
    col[0] = n - 2;
    col[1] = n - 1;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    i        = 0;
    col[0]   = 0;
    col[1]   = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    /*
        Set exact solution; then compute right-hand-side vector.
    */
    PetscCall(VecSet(u, 1.0));
    PetscCall(MatMult(A, u, b));
    KSPCreate(PETSC_COMM_SELF, &ksp);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCAMGX);

    KSPSetOperators(ksp, A, A);

    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);
    //KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);


    KSPSolve(ksp,b,x);
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp,&reason);
    if (reason < 0) {
            std::cout << "Linear solve failed with reason " << KSPConvergedReasons[reason] << std::endl;
        return 1;
    }

    KSPGetIterationNumber(ksp, &its);
        std::cout << "Success! Converged in " << its << " iterations" << std::endl;

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    KSPDestroy(&ksp);

    PetscFinalize();
    return 0;
}
