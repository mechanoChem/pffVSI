# Phase Field Fracture Modeling

## Description

The code we will be using was initially implemented by Siddhartha Srivastava and then updated/corrected using the work from Suh and Sun. This work is all contained in the `*PFF2D*.py` file which loads geometries/meshes from the `mesh_files` directory.

## Log file issues
Since this code uses the Legacy FeniCs environment, there are issues when trying to generate a log file to capture the residual for each Newton iteration within a nonlinear solve. This is because the function for the iterations is pre-compiled C++ code which prints to the command line faster than the JIT compiled Python code. This issue was resolved by using the system flush or setting `flush=True` in the Python `print` statements.

Performance:
The overall run-time as well as attempts to parallelize need to be implemented/tracked in order to benchmark the code.

## Usage
To run this code, make sure you are in an active Docker image (See Installation). Then, update the parameter file based on your desired simulation parameters (details below). Finally, run the command:

```bash
python3 PFF2D_tensile.py params.json
```

If you would like to alter any settings (i.e. number of steps or time step), do so in the `params.json` file.

### Restart

To restart a simulation, simply include the path to the simulation directory:

```bash
python3 PFF2D_tensile.py simulations/simXX/params.json
```

### VSI

Once data is available, the variational system identification may be run using the files labeled `VSI_`, for example:

```bash
python3 VSI_PFF2D_nonlinear.py simulations/simXX/params.json
```

The nonlinear implementation may be run independently, while the linear is run in two steps. First, the matrices are evaluated and saved using:

```bash
python3 VSI_PFF2D_linear.py simulations/simXX/params.json
```

Then, the linear problem is solved using the CVXPY package with:

```bash
python3 VSI_PFF2D_CVX.py simulations/simXX/params.json
```

This was done because the CVXPY package may not be compatible with the legacy FEniCS docker container.

*NOTE*
The system identification files, labeled `VSI_` require the formulation to be the same as the one prescribed in forward `PFF2D`. The `params.json` file helps avoid these discrepancies, but the code must also match. This was not separated out into a helper function due to the limitations when using FEniCS.

### Output 
The code will copy the parameter file to a new simulation folder (unless re-running/restarting an existing simulation). It will then generate a "steps" directory which will save every time step of the simulation as an `h5` file. At the end of the simulation, the load and displacement data for each time step is saved to a text file and if converged, a plot will also be generated. The final `result.h5` file with all steps will be saved to the main simulation directory along with a visualization `result_viz.h5` file.

### Simulation Parameters

Parameters for the simulation, from changes to material parameters to the form of Phase Field Fracture model you would like to run are all contained in the `params.json` file. 

| Parameter | Potential Values | Description |
| ----- | ----- | ----- |
| `read_checkpoint` | true , false | Determines whether or not you want to start from the latest checkpoint file |
| `mesh_name` | string (double_notch.xml, ) | Geometry/Mesh to load, requires the file extension to work |
| `bc_cond` | string (bottom_full, bottom_roller) | Options for the boundary conditions on other surfaces (top is always a vertical displacement prescribed by the `disp_rate` and the current time) |
| `disp_rate` | integer/double value | Rate of load application (mm/s) |
| `dt` | integer/double value | Time step size (s) |
| `num_t_steps` | integer/double value | Maximum number of time steps to run |
| `cg_order` | integer value (0,1,2) | Integration Parameter - Continuous Galerkin Element |
| `q_degree` | integer value | Integration Parameter - Quadrature |
| `mu` | integer/double value | Material shear modulus (MPa) |
| `nu` | integer/double value | Material Poisson ratio (unitless) |
| `zeta` | integer/double value | Material parameter enforcing slope of driving force function, related to d squiggle (unitless) |
| `sig_crit` | integer/double value | Material ultimate stress (MPa) |
| `par_a_1` | integer/double value | Material Anisotropy (unitless) |
| `par_a_2` | integer/double value | Material Anisotropy (unitless) |
| `par_a_3` | integer/double value | Material Anisotropy (unitless) |
| `gamma` | integer/double value | Material Fiber Degree |
| `k1` | integer/double value | Material Anisotropy Parameter (MPa) |
| `k2` | integer/double value | Material Anisotropy Parameter (unitless) |
| `ell` | integer/double value | Damage crack length (mm) |
| `eta` | integer/double value | Damage Rate (MN-s/mm2) |
| `g_c` | integer/double value | Critical energy release rate |
| `psi_cr` | integer/double value | Threshold energy density () |
| `p` | integer/double value | Damage shape parameter |
| `eps_val` | integer/double value | Small epsilon for adding damage to Piola-Kirchoff Tensor |
| `update_type` | integer value ("monolithic", "explicit", "implicit") | Different residual formulations/solution approaches, explicit and implicit are which respect to the history term |
| `hist_fun_type` | integer value ("cpg", "suh") | Different history function evaluations |
| `energy_type` | string ("anisotropic","cnh","svk") | "anisotropic" is the one presented in the Raina and Meihe paper, "cnh" = Compressive Neo-Hookean, and "svk" refers to the formulation presented in the paper from Sun and Suh with SVK stress definition for nonlinear elasticity |
| `degradation_type` | dictionary (keys="quad","quasiQuad") | These correspond to the degredation functions described in the accompanying Overleaf document |
| `geometric_type` | dictionary (keys="linear","singleWell","doubleWell") | These correspond to the geometric crack functions described in the accompanying Overleaf document |
| `dsq_type` | dictionary (keys = "energy","energyCrit","principal","tensile") | Types of history updates, i.e. D squiggle functions. "energy" is a simple energy based criterion that has not worked so far, "energyCrit" refers to the energy criteria from the Suh and Sun paper, "principal" refers to the combination used in Raina and Meihe, and "tensile" is for only using the 22 stress component |
| `pfIndex` | integer (1-6) | Coefficients of kappa and beta based on desired damage model to be used (condensed options, modified for our regularized energy equation, from table of Svolos 2023) |
| `notes` | string | Any type of note to remember to simulation parameters that might be hard coded |

*Note*: Not all paramaters are requires for different simulation setups, however, all parameters are "loaded" at the beginning of the script and must therefore exist in the parameter file.


## Examples

### 1. Double Notch from Suh and Sun (`double_notch.xml`)

This is the mesh from Suh and Sun's paper. The parameters in the default `params.json` file are correct for this example. Note, our version of the code has removed the micropolar contributions and updated from linear to nonlinear elasticity.


## Installation
These codes use FEniCS, there are installation instructions for the legacy version [here](https://fenicsproject.org/download/), however, this version will not guarantee compatibility with this code base. The best way to ensure compatibility is to use a Docker container.

### Docker Container

Pull the latest Docker image:
```
docker pull quay.io/fenicsproject/stable:latest
```

Now, you can launch a container at any time. There is a lot of detail explaining how these commands work in the FEniCS documentation, but for getting started, the command below with launch a detached container. *Note*, the current working directory will open in the container, so navigate to the PFF directory first to make things easier:
```bash
docker run --name fenics -dit -w /home/fenics/shared -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:latest
```

Check all active containers by running the command:
```bash
docker ps
```

Now, attach to the named container with:
```bash
docker exec -it fenics bash
```

To exit the container while leaving it running, enter `ctrl+d` or, if something is running, use `Ctrl+p` and then `Ctrl+q`

**Using GMSH inside of the container**

You can install GMSH the usual way:
```bash
pip install gmsh
```

However if you try to test your installation `python3 -c "import gmsh"` you will get a `libGLU` related error. Install using this:
```bash
apt-get update
apt-get -y install libglu1 libxcursor-dev libxinerama1
```

Install the rest of the requirements:
```bash
pip install meshio h5py
```

*Note*: if you close the container, you may need to go through these installation steps again.