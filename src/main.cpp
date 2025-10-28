#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/PreconditionerFactory.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/models/utils/parametersystem.hpp>
#pragma GCC push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <opm/simulators/linalg/gpuistl/GpuSeqILU0.hpp>
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrixWrapper.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/gpuistl/PreconditionerAdapter.hpp>
#pragma GCC pop
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/ilufirstelement.hh>
#include <opm/simulators/linalg/matrixblock.hh>

#include <fmt/format.h>

#include <boost/program_options.hpp>

#if HAVE_AMGX
#include <amgx_c.h>
#endif

#include "read_binary.hpp"

template <class VectorType>
VectorType
readVector(const std::string& filename)
{
    if (filename.ends_with(".mm")) {
        VectorType vector;
        Dune::loadMatrixMarket(vector, filename);
        return vector;
    } else if (filename.ends_with(".bin")) {
        return readBinaryAMGCLVector<VectorType>(filename);
    } else {
        throw std::runtime_error(
            fmt::format("Unsupported file format: {}.\n\nSupported: .mm, .bin (CASE SENSITIVE!).", filename));
    }
}


template <class MatrixType>
MatrixType
readMatrix(const std::string& filename)
{
    if (filename.ends_with(".mm")) {
        MatrixType matrix;

        // TODO: Can this be done nicer?
        using real = typename MatrixType::block_type::field_type;
        constexpr int blockDim = MatrixType::block_type::rows;
        using MatrixDuneIsExpecting = Dune::BCRSMatrix<Dune::FieldMatrix<real, blockDim, blockDim>>;

        Dune::loadMatrixMarket(reinterpret_cast<MatrixDuneIsExpecting&>(matrix), filename);
        return matrix;
    } else if (filename.ends_with(".bin")) {
        return readBinaryAMGCLMatrix<MatrixType>(filename);
    } else {
        throw std::runtime_error(
            fmt::format("Unsupported file format: {}.\n\nSupported: .mm, .bin (CASE SENSITIVE!).", filename));
    }
}

// Unified solver using FlexibleSolver for both CPU and GPU
// This mirrors the ISTLSolverGPUISTL architecture in OPM Flow
// - Accepts standard solver names: "bicgstab", "loopsolver", "gmres"
// - Same config format works for both CPU and GPU
// - GPU path automatically handles matrix/vector conversion
// - For preconditioner-only benchmarking, use "loopsolver" with "maxiter": 1
template <int dim, class T = double>
std::tuple<unsigned long long, Dune::InverseOperatorResult, bool>
readAndSolve(const std::string& configFilename,
             const std::string& xFilename,
             const std::string& matrixFilename,
             const std::string& rhsFilename,
             const std::string& accelerator)
{
    using M = Opm::MatrixBlock<T, dim, dim>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using CPUVector = Dune::BlockVector<Dune::FieldVector<T, dim>>;

    Opm::PropertyTree configuration(configFilename);

    auto B = readMatrix<SpMatrix>(matrixFilename);
    auto x = readVector<CPUVector>(xFilename);
    auto rhs = readVector<CPUVector>(rhsFilename);

    Dune::InverseOperatorResult result;
    bool failed = false;
    unsigned long long duration_us = 0;

    if (accelerator == "cpu") {
        using CPUOperator = Dune::MatrixAdapter<SpMatrix, CPUVector, CPUVector>;
        using CPUFlexibleSolver = Dune::FlexibleSolver<CPUOperator>;

        auto wc = []() -> CPUVector {
            throw std::runtime_error("getQuasiImpesWeights is not supported in the benchmarking library.");
            return CPUVector();
        };

        try {
            auto BOperator = std::make_shared<CPUOperator>(B);
            auto solver = CPUFlexibleSolver(*BOperator, configuration, wc, 0);

            auto start = std::chrono::high_resolution_clock::now();
            solver.apply(x, rhs, result);
            auto end = std::chrono::high_resolution_clock::now();

            duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) {
            std::cerr << "CPU solver failed: " << e.what() << "\n";
            failed = true;
        }
    } else if (accelerator == "gpu") {
        using GPUMatrix = Opm::gpuistl::GpuSparseMatrixWrapper<T>;
        using GPUVector = Opm::gpuistl::GpuVector<T>;
        using GPUOperator = Dune::MatrixAdapter<GPUMatrix, GPUVector, GPUVector>;
        using GPUFlexibleSolver = Dune::FlexibleSolver<GPUOperator>;

        auto wc = []() -> GPUVector {
            throw std::runtime_error("getQuasiImpesWeights is not supported in the benchmarking library.");
            return GPUVector(0);
        };

        try {
            // Convert matrix to GPU
            auto BonGPU = GPUMatrix::fromMatrix(B);
            auto BOperator = std::make_shared<GPUOperator>(BonGPU);

            // Create FlexibleSolver
            auto solver = GPUFlexibleSolver(*BOperator, configuration, wc, 0);

            // Convert vectors to GPU
            auto xOnGPU = GPUVector(x);
            auto rhsOnGPU = GPUVector(rhs);

            auto start = std::chrono::high_resolution_clock::now();
            solver.apply(xOnGPU, rhsOnGPU, result);
            OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();

            duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) {
            std::cerr << "GPU solver failed: " << e.what() << "\n";
            failed = true;
        }
    } else {
        throw std::runtime_error("Invalid accelerator: " + accelerator);
    }

    return std::make_tuple(duration_us, result, failed);
}


void
printResults(const std::string& accelerator,
             unsigned long long runtime_us,
             const Dune::InverseOperatorResult& result,
             bool failed)
{
    boost::property_tree::ptree tree;
    tree.add("accelerator", accelerator);
    tree.add("runtime_us", runtime_us);
    tree.add("failed_by_exception", failed);
    tree.add("iterations", result.iterations);
    tree.add("reduction", result.reduction);
    tree.add("converged", result.converged);
    tree.add("conv_rate", result.conv_rate);
    tree.add("elapsed", result.elapsed);
    tree.add("condition_estimate", result.condition_estimate);

    boost::property_tree::write_json(std::cout, tree, true);
}


int
main(int argc, char** argv)
{
    [[maybe_unused]] const auto& helper = Dune::MPIHelper::instance(argc, argv);

#if HAVE_AMGX
    AMGX_SAFE_CALL(AMGX_initialize());
#endif

    // Register OPM parameters that preconditioners might need
    // These are Flow parameters that AMGX and other preconditioners may access
    // Default values are taken from the parameter structs (e.g., CprReuseInterval::value = 30)
    Opm::Parameters::Register<Opm::Parameters::CprReuseInterval>
        ("Reuse preconditioner interval");

    // Close parameter registration - required before retrieving any parameters
    Opm::Parameters::endRegistration();

    namespace po = boost::program_options;
    po::options_description desc("OPM Linear Solver Benchmarking Tool");
    desc.add_options()("help,h", "Produce this help message")(
        "matrix-file,m", po::value<std::string>()->required(), "Matrix filename (.mm or .bin)")(
        "initial-guess-file,x", po::value<std::string>()->required(), "Initial guess filename")(
        "rhs-file,y", po::value<std::string>()->required(), "Right-hand side filename")(
        "configfile", po::value<std::string>()->required(), "Solver configuration file (.json)")(
        "linear-solver-accelerator",
        po::value<std::string>()->default_value("cpu"),
        "Linear solver accelerator: 'cpu' or 'gpu' (default: cpu)")(
        "block-size,b", po::value<size_t>(), "Block size (required for binary files)");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n\n";
            std::cout << "Example usage:\n";
            std::cout << "  " << argv[0] << " -m matrix.mm -x init.mm -y rhs.mm --configfile config.json\n";
            std::cout
                << "  " << argv[0]
                << " -m matrix.mm -x init.mm -y rhs.mm --configfile config.json --linear-solver-accelerator gpu\n";
            return EXIT_SUCCESS;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        std::cerr << desc << "\n";
        return EXIT_FAILURE;
    }

    // Get command-line arguments
    const auto matrixFilename = vm["matrix-file"].as<std::string>();
    const auto xFilename = vm["initial-guess-file"].as<std::string>();
    const auto rhsFilename = vm["rhs-file"].as<std::string>();
    const auto configFilename = vm["configfile"].as<std::string>();
    const auto accelerator = vm["linear-solver-accelerator"].as<std::string>();

    // Validate accelerator
    if (accelerator != "cpu" && accelerator != "gpu") {
        std::cerr << "Error: Invalid accelerator '" << accelerator << "'. Must be 'cpu' or 'gpu'.\n";
        return EXIT_FAILURE;
    }

    // Determine block size
    size_t dim = 2;
    if (vm.count("block-size")) {
        dim = vm["block-size"].as<size_t>();
    } else if (matrixFilename.ends_with(".mm")) {
        std::ifstream matrixfile(matrixFilename);
        std::string line;
        const std::string lineToFind = "% ISTL_STRUCT blocked";
        while (std::getline(matrixfile, line)) {
            if (line.substr(0, lineToFind.size()) == lineToFind) {
                dim = std::atoi(line.substr(lineToFind.size() + 2).c_str());
                break;
            }
        }
    } else {
        std::cerr << "Error: For binary files, you must specify --block-size\n";
        return EXIT_FAILURE;
    }

    // Run solver based on block size
    try {
        std::tuple<unsigned long long, Dune::InverseOperatorResult, bool> result;

        switch (dim) {
        case 1:
            result = readAndSolve<1>(configFilename, xFilename, matrixFilename, rhsFilename, accelerator);
            break;
        case 2:
            result = readAndSolve<2>(configFilename, xFilename, matrixFilename, rhsFilename, accelerator);
            break;
        case 3:
            result = readAndSolve<3>(configFilename, xFilename, matrixFilename, rhsFilename, accelerator);
            break;
        case 4:
            result = readAndSolve<4>(configFilename, xFilename, matrixFilename, rhsFilename, accelerator);
            break;
        default:
            std::cerr << "Error: Unsupported block dimension " << dim << "\n";

#if HAVE_AMGX
            AMGX_SAFE_CALL(AMGX_finalize());
#endif

            return EXIT_FAILURE;
        }

        auto [runtime_us, solve_result, failed] = result;
        printResults(accelerator, runtime_us, solve_result, failed);

#if HAVE_AMGX
        AMGX_SAFE_CALL(AMGX_finalize());
#endif

        return failed ? EXIT_FAILURE : EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";

#if HAVE_AMGX
        AMGX_SAFE_CALL(AMGX_finalize());
#endif

        return EXIT_FAILURE;
    }
}
