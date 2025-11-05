# opm-linear-solver-lab
Experimental playground for testing out linear solvers in OPM Flow.

## Linear System in OPM Flow

OPM Flow uses a Newton-Raphson method for solving reservoir simulation equations. At each Newton iteration, a linear system is solved:

`J(yn) Δy = -R(yn)`

where:

- `J(yn)` is the Jacobian matrix — derivatives of residual equations with respect to primary variables
- `-R(yn)` is the right-hand side (rhs) — the negative of the residual vector evaluated at the current Newton state
- `Δy` is the correction/update vector — not the full solution

The correction vector `Δy` is initialized to zero before calling the iterative solver, which computes the update to apply: `yn+1 = yn + Δy`.

**Note on wells**: By default, OPM Flow uses a Schur complement approach to handle well equations separately from the reservoir system. This tool requires the well contributions to be explicitly included in the Jacobian matrix, which can be enabled in opm flow with `--matrix-add-well-contributions=true`. This ensures the benchmarked system matches the complete linearized problem.

## Compiling
We assume you have opm and dune in your prefix path. Compiling should just be

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Note that you probably want to specify `CMAKE_PREFIX_PATH` to point to the locations of the OPM and Dune build folder, that is

```bash
# from build.
# IMPORTANT: Notice the quotation marks around the prefix path.
# This is IMPORTANT when you have multiple paths
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/path/to/opm-common/build;/path/to/opm-grid/build;MOREPATHSHERE"
```

To simplify this process, we've added the script `build_helpers/prefix_path.sh` to generate this prefix path. If you directory structure looks like this:

```bash
#DUNE
/some/path/for/dune/dune-common/build
/some/path/for/dune/dune-grid/build
/some/path/for/dune/dune-geometry/build
/some/path/for/dune/dune-istl/build

#OPM
/some/path/other/path/for/opm/opm-common/build
/some/path/other/path/for/opm/opm-grid/build
/some/path/other/path/for/opm/opm-models/build
/some/path/other/path/for/opm/opm-simulators/build
```

you can run cmake as

```bash
# from build.
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(bash ../build_helpers/prefix_path.sh /some/path/other/path/for/opm/ /some/path/for/dune/dune-common/)"
```

### Note on libfmt

You probably need to have libfmt installed. On Ubuntu this can be accomplished by installing

```bash
sudo apt install libfmt-dev
```

Alternatively you can manually donwload and install libfmt. Make sure to extend your `CMAKE_PREFIX_PATH` to contain the install directory of libfmt.
## Running

The tool uses a unified interface similar to OPM Flow's `--linear-solver-accelerator` option.

### Configurations

- **`configurations/`**: Standard configs using `"bicgstab"`, `"dilu"`, etc. - work for both CPU and GPU
- **`configurations/solver_adapter/`**: Legacy configs using `"gpubicgstab"`, `"gpudilu"`, etc. - for SolverAdapter approach

### GPU Execution Modes

There are **two ways** to run on GPU, with different performance characteristics:

#### 1. FlexibleSolver (Recommended - Full GPU)
Runs **both solver and preconditioner on GPU**. Uses standard configs from `configurations/`:

```bash
# Same config works for CPU...
./linsolverlab \
    -m ../examples/matrices/spe1/matrix.mm \
    -r ../examples/matrices/spe1/rhs.mm \
    --configfile ../examples/configurations/dilu.json

# ...and GPU (everything on GPU)
./linsolverlab \
    -m ../examples/matrices/spe1/matrix.mm \
    -r ../examples/matrices/spe1/rhs.mm \
    --configfile ../examples/configurations/dilu.json \
    --linear-solver-accelerator gpu
```

- **Architecture**: `FlexibleSolver` with `GpuSparseMatrixWrapper` (mirrors `ISTLSolverGPUISTL` in OPM Flow)
- **Solver names**: `"bicgstab"`, `"loopsolver"`, `"gmres"`
- **Preconditioner types**: `"dilu"`, `"ilu0"`, etc. (automatically run on GPU)
- **Use case**: Best for full GPU acceleration
- **Note**: For preconditioner-only benchmarking, use `"loopsolver"` with `"maxiter": 1`

#### 2. SolverAdapter (Legacy)
Uses configs from `configurations/solver_adapter/` subfolder:

```bash
./linsolverlab \
    -m ../examples/matrices/spe1/matrix.mm \
    -r ../examples/matrices/spe1/rhs.mm \
    --configfile ../examples/configurations/solver_adapter/gpudilu.json
```

- **Architecture**: `SolverAdapter` wrapping GPU preconditioners
- **Solver names**: `"gpubicgstab"`
- **Preconditioner types**: `"gpudilu"`, `"gpuilu0"`, `"gpudilu"`
- **Use case**: Legacy SolverAdapter pattern


### Command-Line Options

```
  -h, --help                        Show help message
  -m, --matrix-file arg             Matrix filename (.mm or .bin)
  -r, --rhs-file arg                Right-hand side filename
  -w, --cpr-weights-file arg        CPR weights filename (.mm or .bin) - optional
  --configfile arg                  Solver configuration file (.json)
  --linear-solver-accelerator arg   'cpu' (default) or 'gpu'
  -b, --block-size arg              Block size (required for binary files)
```

### CPR Preconditioners

For CPR preconditioners, provide the weights file using `--cpr-weights-file`:

```bash
./linsolverlab \
    -m matrix.mm \
    -r rhs.mm \
    -w weights.mm \
    --configfile cpr_config.json
```

The weights file should be in MatrixMarket format.

### Example Output

The tool outputs JSON with timing and convergence information:

```json
{
    "accelerator": "cpu",
    "precond_update_time_s": "0.0012345",
    "solver_apply_time_s": "0.0098765",
    "solver_total_time_s": "0.011111",
    "failed_by_exception": "false",
    "iterations": "24",
    "reduction": "8.0168234269055139e-13",
    "converged": "true",
    "conv_rate": "0.31332864211998734",
    "solver_internal_elapsed_s": "0.010678536000000001",
    "condition_estimate": "-1"
}
```

**Timing Fields**:
- `precond_update_time_s`: Time spent in `preconditioner.update()`, which is the time spent in the preconditioner to update it based on the current matrix values.
- `solver_apply_time_s`: Time spent in `solver.apply()` (the actual linear solve)
- `solver_total_time_s`: Total measured time (`precond_update_time_s` + `solver_apply_time_s`)
- `solver_internal_elapsed_s`: Dune's internal timing from the solver (may differ slightly from external measurements)


## Benchmarking Only the Preconditioner

You can benchmark preconditioners by using `"loopsolver"` with `"maxiter": 1`:

```bash
./linsolverlab \
    -m ../examples/matrices/spe1/matrix.mm \
    -r ../examples/matrices/spe1/rhs.mm \
    --configfile config_preconditioner_only.json \
    --linear-solver-accelerator gpu
```

Example config (`config_preconditioner_only.json`):
```json
{
    "tol": "1e-12",
    "maxiter": "1",
    "verbosity": "0",
    "solver": "loopsolver",
    "preconditioner": {
        "type": "dilu"
    }
}
```

**Note**: `loopsolver` with `maxiter=1` applies the preconditioner exactly once, making it ideal for preconditioner benchmarking.


## Running benchmarks


### OPM-flow


Generating benchmark data

```python
# usage: run_opm_benchmark.py <subfolder1> [<subfolder2> ...]
python benchmarking_scripts/run_opm_benchmark.py sleipner
```

Adjusting the experiment in the file

```python
    json_files = [
        'examples/configurations/cpu/ilu0.json --block-size 2',
        'examples/configurations/cpu/dilu.json --block-size 2',
        'examples/configurations/gpu/gpuilu0.json --block-size 2',
        'examples/configurations/gpu/gpuopmilu0.json --block-size 2',
        'examples/configurations/gpu/gpudilu.json --block-size 2'
    ]
    matrix_root = "./examples/matrices"  # folder where the matrix files can be found
    limit = 10  # Adjust this to change the number of matrices processed per directory
```



Plotting the results

```python
python benchmarking_scripts/process_and_plot_opm.py
```
