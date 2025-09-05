# --------------------------------
# VSI: Variational System Identification
# PFF: Phase Field Fracture
# Viscoelastic crack growth with nonlinear elasticity
#
# Author: Siddhartha Srivastava (sidsriva@umich.edu) and Liz Livingston (lizliv@umich.edu)
# Date: Aug 29, 2023
# Updated: Sep 1, 2025
# See PFF2d_tensile_staggered.py for forward solve

# Usage:
#     python3 VSI_PFF2D_linear.py simulations/simXX/params.json
# --------------------------------
from ufl import *
from fenics import *
import numpy as np
import os
import sys
import argparse
import json
import shutil
import pandas as pd

import principal_components as pc

set_log_active(True) # Dolfins logging setup

import logging
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

parameters['linear_algebra_backend'] = 'Eigen'

import time


def setupSimulationFiles(param_file):
    
    # Handle the parameter file to get the simulation path and parameter file name
    separate_path = False
    if '/' in param_file:
        sep = '/'
        separate_path = True
    elif '\\' in param_file:
        sep = '/'
        separate_path = True
        
    # Procedure if this is a new simulation (no separator --> i.e. a new parameter file)
    if separate_path == False:
        sim_num = getSimNum('simulations','',False)
        print(f'Found {sim_num} simulations')
        load_path = os.path.join('simulations',f'sim{sim_num:02}')
        simExists = True
        while simExists:
            if os.path.exists(load_path):
                print(f'Simulation {sim_num} directory exists, incrementing by 1')
                sim_num = sim_num + 1
                load_path = os.path.join('simulations',f'sim{sim_num:02}')
            else:
                simExists = False
        os.mkdir(load_path)
        shutil.copyfile(param_file, os.path.join(load_path,param_file))
        print(f'Created directory for load path and copied parameter file')
    # Handle a parameter file given with a provided path
    elif separate_path == True:
        temp = param_file.split(sep)
        load_path = ''
        for i in range(len(temp)-1):
            load_path = os.path.join(load_path,temp[i])
        param_file = temp[-1]
        sim_num = getSimNum(load_path,sep,True)
        print('Path separator detected, splitting load path from file name')
    return load_path,param_file
def getSimNum(dir,sep,existingSim):
    # Logic check: if opening an existing sim, the simulations directory already exists, this is only necessary if starting on a new machine
    if os.path.exists(dir) == False and existingSim == False:
        print(f'WARNING: No simulation directory exists, creating a simulations directory here: {os.getcwd()}')
        os.makedirs(dir)
    # If a new simulation, get the number of previously run simulations
    if existingSim == False:
        # Save all simulations to a "simulations" directory. label is just an index, parameter file should have simulation details
        sim_dirs = [ f.path for f in os.scandir(dir) if f.is_dir() ]
        sim_num = 0
        for dirname in list(sim_dirs):
            dirname = os.path.basename(os.path.normpath(dirname))
            if 'sim' in dirname and len(dirname) == 5:
                    sim_num = sim_num + 1
    # If an existing simulation, get the number from the path passed in as an argument
    else:
        dir = dir.split(sep)
        if len(dir) > 2:
            print('Unexpected directory structure, edit the getSimNum() function')
            exit()
        else:
            sim_num = int(dir[1].split('sim')[1])
    
    return sim_num


#Basis set for degradation function

def degradation_quadratic():
    loc_g_d         = pow(one-d,2)
    return loc_g_d
def degradation_deriv_quadratic():
    loc_g_d_prime   = -2*(one-d)
    return loc_g_d_prime

def degradation_quasi_quadratic():
    p=10 #Fixing m and p based on Sun's paper
    m=50 
    loc_g_d = ((1.-d)**2)/((1.-d)**2 + m*d*(1.+p*d))
    return loc_g_d
def degradation_deriv_quasi_quadratic():
    p=10 #Fixing m and p based on Sun's paper
    m=50 
    loc_g_d_prime = ((d-1.)*(d*(2.*p+1.) + 1.)*m)/(((d**2)*(m*p+1.) + d*(m-2.) + 1.)**2)
    return loc_g_d_prime

#Basis set for geometric function

def geometric_linear():
    return d
def geometric_deriv_linear():
    return one

def geometric_single_well():
    return pow(d,2)
def geometric_deriv_single_well():
    return 2*d

def geometric_double_well():
    return d*d*(one-d*d)
def geometric_deriv_double_well():
    return 2*d*(one-d*d) - 2*d*d*d


# Basis set for D_squiggle function
def dsq_energy():
    #Energy based damage evolution criteria
    return psi_e_pos
def dsq_energyCrit():
    # For now this needs psi critical
    psi_i = psi_e - psi_crit
    return psi_crit + (1./2.)*(psi_i + abs(psi_i))
def dsq_principal():
    # Max principal stress based criteria
    # For now this needs sigma critical
    lam1, lam2 = pc.eigenValues(cauchy_stress,tol=1e-10)
    lam1 = (lam1 + abs(lam1))/2         # Only include positive stretches for tensile stress based criterion 
    lam2 = (lam2 + abs(lam2))/2
    _func = pow(lam1/sig_crit, 2) + pow(lam2/sig_crit, 2) - 1 # Anisotropic setting
    return 0.5*(_func + abs(_func))
def dsq_sigma22():
    # For now this needs sigma critical    
    e2_dir = Constant((0.,1.))
    _sig22 = dot(e2_dir,dot(e2_dir, cauchy_stress))
    sig_22 = (_sig22 + abs(_sig22))/2
    _func = pow(sig_22/sig_crit,2) -  1 # No anisotropy and not really isotropic either

    return 0.5*(_func + abs(_func))


# Driving force -----------------------------------
def hist(dsq,H_old):
    return conditional(ge(dsq, H_old), dsq, H_old)
def H(H_old):
  
  psi_i_new = psi_e - psi_crit
  psi_i_old = psi_e_n - psi_crit
  
  psi_new =  psi_crit + (1./2.)*(psi_i_new + abs(psi_i_new))
  psi_old =  psi_crit + (1./2.)*(psi_i_old + abs(psi_i_old))

  return conditional(lt(psi_old, psi_new), psi_new, H_old)

# Macaulay Bracket Functions
def mac_pos(A):
    return (abs(A)+A)/2
def mac_neg(A):
    return (abs(A)-A)/2

degradation_basis = {'quad': {'name': 'quadratic', 'func':degradation_quadratic, 'dfunc': degradation_deriv_quadratic}, 
                     'quasiQuad': {'name': 'quasiQuadratic', 'func':degradation_quasi_quadratic, 'dfunc': degradation_deriv_quasi_quadratic},
                     }
geometric_basis = {'linear': {'name': 'linear', 'func':geometric_linear, 'dfunc': geometric_deriv_linear}, 
                     'singleWell': {'name': 'singleWell', 'func':geometric_single_well, 'dfunc': geometric_deriv_single_well},
                     'doubleWell': {'name': 'doubleWell', 'func':geometric_double_well, 'dfunc': geometric_deriv_double_well},
                     }
history_basis = {'energy': {'name': 'energy', 'func':dsq_energy, },
                }

num_alpha = len(degradation_basis.keys())
num_beta = len(geometric_basis.keys())+1 #+1 for the gradient energy term
num_zeta = len(history_basis.keys())

# ------------------------------------------------
#                   Parse
# ------------------------------------------------
parser = argparse.ArgumentParser(description='Assign simulation parameters and select solver type.')
parser.add_argument('paramFile', metavar='N', type=str,
                    help='Name or path to parameter file')

parser.add_argument('-noise','--add_noise', help="Flag to add noise", required=False, action='store_true')
parser.add_argument('-nf', '--noise_field', help="Noise field to be applied: ux, uy, or d", required=False, default="uy")
parser.add_argument('-ns', '--noise_sigma', help="Noise standard deviation (sigma) to be applied: i.e. 5e-7", required=False, default=5e-7)

parser.add_argument('-subsample','--subsample', help="Flag to subsample", required=False, action='store_true')
parser.add_argument('-ss', '--sample_size', help="Subsampling to be applied: i.e. 2 or 4", required=False, default=2)
args = parser.parse_args()

param_file = args.paramFile


# ------------------------------------------------
#                   VSI SETTINGS
# ------------------------------------------------
# Indices needed for reading the load_curve file
DTINDEX = 2
TINDEX = 1

# Settings that change the way VSI is performed
LOAD_COARSE = False
coarse_mesh_idx = 'M1'
coarse_mesh_name = f'doublenotch_coarsen_{coarse_mesh_idx}.xdmf'

ADD_NOISE = False
noise_field = args.noise_field
noise_mu = 0
noise_sigma = float(args.noise_sigma)
if args.add_noise:
    ADD_NOISE = True
    print(f'Adding noise on {noise_field} DOF with sigma of {noise_sigma}')

THRESHOLD_NOISE = False

OVERWRITE_START_END = False # Set to true if you want to set new_start_index and new_num_t_steps
new_start_index = 0
new_num_t_steps = 740

SUBSAMPLE_TIME = False
subsample_size = int(args.sample_size)
if args.subsample:
    SUBSAMPLE_TIME = True
    print(f'Will subsample time steps every {subsample_size} steps')
    
OVERWRITE_QUAD_DEGREE = False
new_quad_degree = 20

if OVERWRITE_START_END == False:
    save_name = 'full'
elif OVERWRITE_START_END == True:
    save_name = f'{new_num_t_steps}'

if ADD_NOISE == True:
    save_name += f'-noise_{noise_field}'
if SUBSAMPLE_TIME == True:
    save_name += f'-sample_{subsample_size}'
print(f'Files will be saved using the name: {save_name}')

# #------------------------------------------------
# #          Define Parameters
# #------------------------------------------------

print('Current Directory:',os.getcwd())

load_path,param_file = setupSimulationFiles(param_file)
print(f'     Load path is: {load_path}')
print(f'     Parameter file is: {param_file}\n')
param_path = os.path.join(load_path,param_file)
if LOAD_COARSE:
    save_path = os.path.join(load_path,f'vsi-coarse_{coarse_mesh_idx}')
else:
    save_path = os.path.join(load_path,'vsi')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

if ADD_NOISE:
    save_path = os.path.join(save_path,f'noise_{noise_field}_{noise_sigma:1.0e}')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # Get number of folders within noisy subdir
    noisySims = [x[0] for x in os.walk(save_path) if os.path.isdir(x[0])]
    save_path = os.path.join(save_path,f'noise-{len(noisySims):02d}')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    else:
        print(f'Check length of noisy sim directories, tried to save in {len(noisySims)}')
        exit()
elif LOAD_COARSE:
    save_path = os.path.join(save_path,f'coarse_{coarse_mesh_idx}')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
elif SUBSAMPLE_TIME:
    save_path = os.path.join(save_path,f'subsample_{subsample_size}')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

print(f'     Save path is: {save_path}')

# This will load the parameters and then dynamically generate variables from the keys of the json file
with open(param_path, "r") as read_file:
    params = json.load(read_file)


# Mesh/Geom information
if LOAD_COARSE:
    mesh_name = coarse_mesh_name
else:
    mesh_name = params['mesh_name']
# Loading conditions
read_checkpoint = params['read_checkpoint']
bc_cond = params['bc_cond']
# Time step
dt = params['dt']
tau = Constant(dt)
num_t_steps = int(params['num_t_steps'])
# Integration
cg_order = params['cg_order']
q_degree = params['q_degree']
# Elasticity Parameters
mu, nu  = params['mu'], params['nu'] # MPa, unitless
gamma   = params['gamma']            # Fiber degree
k1, k2  = params['k1'], params['k2']    # MPa, unitless
sig_crit = params['sig_crit']     # unitless, MPa 
par_a_1, par_a_2, par_a_3    = params['par_a_1'], params['par_a_2'], params['par_a_3']    # All unitless, 0.7 for the first coordinate
# Damage Parameters
ell = params['ell'] # mm
eta = params['eta'] # 1e-5 MN-s/mm2
gc = params['g_c']
psi_crit = params['psi_cr']
eps_val = params['eps_val']  # Small epsilon for adding damage to Piola-Kirchoff Tensor
# Solver Parameters
max_subiter = params['max_subiter'] # 100
err_tol = params['err_tol_subiter'] # 1E-6
update_type = params['update_type']
hist_fun_type = params['hist_fun_type']
energy_type = params['energy_type']
# Damage Parameters
degradation_type = params['degradation_type']
geometric_type = params['geometric_type']
dsq_type    = params['dsq_type']
pfIndex = params['pfIndex']

# if/else statement determine f_c, coefficient for beta, coefficient for zeta
if pfIndex==1:
    f_c = Constant(2.*ell)
    pfbeta = Constant(1/(8.*ell*ell))
elif pfIndex==2:
    f_c = Constant(2.)
    pfbeta = Constant(1/(4.))
elif pfIndex==3:
    f_c = Constant(ell)
    pfbeta = Constant(1/(2.*l*l))
elif pfIndex==4:
    f_c = Constant(2.*l/(np.pi))
    pfbeta = Constant(1/(2*ell*ell))
elif pfIndex==5:
    f_c = Constant(3*ell/(4))
    pfbeta = Constant(1/(2*ell*ell))
elif pfIndex==6:
    f_c = Constant(3*ell/(4*np.sqrt(2)))
    pfbeta = Constant(1/(ell*ell))
else:
    print(f'ERROR: pfIndex of {pfIndex} does not exist')
    exit()

pfzeta = Constant(1/(gc*f_c))


#-------------------------------------------------
#           Basic Loading/printout
#-------------------------------------------------
if os.path.exists(os.path.join(load_path,'load_curve.txt')) == False:
    print(f"No restart files ({os.path.join(load_path,'load_curve.txt')}) exist, cannot read checkpoint\n")
    read_checkpoint = False
    exit()
    
# This is not a simulation restart, start should be 0
start_index = 0
if read_checkpoint==True:
    load_curve = np.loadtxt(os.path.join(load_path,'load_curve.txt'), delimiter=' ')
    end_index = len(load_curve[:,0])-1
    print(f"Loaded load_curve...\n     Simulation has: {end_index} steps")
# If num_t_steps in the parameter file is larger than the number of steps in the simulation, overwrite the end index
if end_index < num_t_steps:
    num_t_steps = end_index

if OVERWRITE_START_END == True:
    start_index = new_start_index
    num_t_steps = new_num_t_steps
    if start_index + num_t_steps <= end_index:
        end_index = start_index + num_t_steps
    else:
        print(f'End index with custom settings is too big, changing start index')
        start_index = end_index - num_t_steps

# The load_step_arr will always start at 0 (for loading history) while the step_save_arr is the indices of steps to be saved
if SUBSAMPLE_TIME == False:
    load_step_arr = np.arange(0,end_index+1)
elif SUBSAMPLE_TIME == True:
    load_step_arr = np.arange(0,end_index+1,subsample_size)
step_idx = np.where(load_step_arr>=start_index)
step_save_arr = load_step_arr[step_idx]

start_index = step_save_arr[0]
end_index = step_save_arr[-1]
num_t_steps = len(step_save_arr)
if SUBSAMPLE_TIME == True:
    print(f'     Inferring every {subsample_size} steps from {start_index} to {end_index} ({num_t_steps} time steps)\n')
    print('       For reference, the steps are:', load_step_arr)
else:
    print(f'     Inferring from {start_index} to {end_index} ({num_t_steps} time steps)\n')


if OVERWRITE_QUAD_DEGREE:
    print(f'Forward run with quadrature {q_degree}, Inferring using quadrature {new_quad_degree}')
    q_degree = new_quad_degree

# ---------------------------------------------------------------- 
# Read mesh
# ----------------------------------------------------------------
# Define mesh
from dolfin import *

# Process the mesh file extension
print(f'Loading the mesh: {mesh_name}')
mesh_type = mesh_name.split('.')
if len(mesh_type) < 2:
    print('ERROR: Did not include file extension in mesh_name, please update parameter file')
    exit()
else:
    mesh_type = mesh_type[1]

# Check that the mesh actually exists
mesh_path = f'./mesh_files/{mesh_name}'
if os.path.exists(mesh_path) == False:
    print(f'ERROR: Mesh does not exist, please check file name/path in parameter file: {mesh_path}')
    exit()

# Load the Mesh
if mesh_type == 'xml':
    print('Loading xml file')
    mesh = Mesh('./mesh_files/'+mesh_name)
elif mesh_type == 'xdmf':
    print('Loading xdmf file')
    mesh = Mesh()
    with XDMFFile(f"./mesh_files/{mesh_name}") as file:
        file.read(mesh)

dx_ = dx(metadata={'quadrature_degree': q_degree})
x = SpatialCoordinate(mesh)
ndim = mesh.topology().dim()

#------------------------------------------------
#          Define Space
#------------------------------------------------
P_u = VectorElement('CG', mesh.ufl_cell(), cg_order) #For displacements
P_d = FiniteElement('CG', mesh.ufl_cell(), cg_order-1) #For damage
V_u = VectorFunctionSpace(mesh, 'CG', cg_order)
V_h = FunctionSpace(mesh, 'DG', cg_order-1)

element = MixedElement([P_u,P_d]) 
V = FunctionSpace(mesh, element)

# Define test and trial functions
w           = TestFunction(V)
w_u, w_d    = split(w)

# Define functions for variables
state       = Function(V)
state_n     = Function(V)
state_temp  = Function(V)   # For implicit integration scheme

#Need to track history for all basis:
H_n = []
H_temp = []
for zeta in range(num_zeta):
    H_n.append(Function(V_h)) # Need a scalar field for storing history
    H_temp.append(Function(V_h)) # Need a scalar field for storing history of previous time step

# Split system functions to access components
u, d            = split(state)      # u.split(True) for postprocessing 
u_n, d_n        = split(state_n)    # u_n.split(True)for postprocessing
# u_temp, d_temp  = split(state_temp) # u_n.split(True)for postprocessing

# Gradient: 
grad_w_d    = grad(w_d)
grad_d      = grad(d) 

#------------------------------------------------
#            Useful functions
#------------------------------------------------
unit_y_normal = Constant((0.0, 1.0))
one     = Constant(1.0)
zero    = Constant(0.0)
zero_vector = Constant([0.0]*ndim)
zero_state  = Function(V)

#------------------------------------------------
#          Define Boundary and Loading condition
#------------------------------------------------
dof_coordinates = V_u.tabulate_dof_coordinates()                    
dof_coordinates.resize((V_u.dim(), mesh.geometry().dim()))                           
dof_x = dof_coordinates[:, 0] 
dof_y = dof_coordinates[:, 1]

spec_height = dof_y.max()-dof_y.min()

#x-(horizontal), y-(vertical) 
top     = CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.max())
bottom  = CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.min())
left    = CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.min())
right   = CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.max())

#Get BC for current time step
# NOTE: Dirichlet BCs need to be 0 in Weak Form (i.e. in the residual)
BC_u_top            = DirichletBC(V.sub(0).sub(1), zero, top)
BC_u_bottom_full    = DirichletBC(V.sub(0), Constant((0.0,0.0)), bottom)
BC_u_bottom         = DirichletBC(V.sub(0).sub(1), zero, bottom)
BC_u_right          = DirichletBC(V.sub(0).sub(0), zero, right)
BC_u_left_corner    = DirichletBC(V.sub(0).sub(0), zero, "x[0]<0.0+DOLFIN_EPS && x[1]<0.0+DOLFIN_EPS", method="pointwise")

if bc_cond == 'bottom_full':
    bcs = [BC_u_top, BC_u_bottom_full]
elif bc_cond == 'bottom_roller':
    # bcs = [BC_u_top, BC_u_bottom]
    bcs = [BC_u_top, BC_u_bottom,BC_u_left_corner]

#Estimate boundary dofs
temp_func = Function(V)
temp_func.vector()[:]=2
for bc in bcs:
    bc.apply(temp_func.vector())
interior_dofs = np.flatnonzero(temp_func.vector().get_local() == 2).astype(np.int32)
boundary_dofs = np.flatnonzero(temp_func.vector().get_local() == 0).astype(np.int32)
ndof = len(interior_dofs)
print('Number of dof', ndof)
#------------------------------------------------
#            Constitutive Relationships
#------------------------------------------------
#Elasticity
I = Identity(ndim)          # Identity Tensor
F = I + grad(u)             # Deformation Gradient
F_n = I + grad(u_n)             # Deformation Gradient

JF  = det(F)
FiT = transpose(inv(F))
Ce = F.T*F                # Right Cauchy-Green Tensor
E_gr = 0.5*(Ce-I)        # Right Cauchy-Green Strain Tensor

E_gr1,E_gr2 = pc.eigenValues(E_gr,tol=1e-15)

beta = 2*nu/(1 - 2*nu)
lmbda = Constant((2*mu*nu)/(1 - 2*nu))

# 4th invariants for Anisotropic tensor
a1 = Constant((np.cos(np.deg2rad(gamma)), -np.sin(np.deg2rad(gamma))))
a2 = Constant((np.cos(np.deg2rad(gamma)), np.sin(np.deg2rad(gamma))))
M1 = outer(a1,a1)
M2 = outer(a2,a2)
CM1 = tr(Ce.T*M1)
CM2 = tr(Ce.T*M2)
Fa1a1 = outer(dot(F,a1), a1)
Fa2a2 = outer(dot(F,a2), a2)

# Invariants of Deformation Tensors
Ic = tr(Ce)
Jc  = det(Ce)
invC = inv(Ce)

# Stress and Elastic Energy
if energy_type == 'anisotropic':
    psi_e = (mu/beta)*(pow(JF,-beta)-1) + 0.5*mu*(inner(F,F)-2) # Raina 2016 (Eq 36)
    # Anisotropic energy components, Raina 2016 (Eq 37)
    if k2 != 0:
        psi_e += (k1/k2) * ( exp(k2*(CM1 - 1)) - 1) - k1*(CM1 - 1)
        psi_e += (k1/k2) * ( exp(k2*(CM2 - 1)) - 1) - k1*(CM2 - 1)
    P = -mu*pow(JF,-beta)*FiT + mu*F       #Nonlinear elastic stress without damage
    P += 2*k1*(exp(k2*(CM1 - 1))-1)*Fa1a1  #Anisotropy
    P += 2*k1*(exp(k2*(CM2 - 1))-1)*Fa2a2
elif energy_type == 'cnh':
    psi_e = 0.25*lmbda*(Jc-1) - 0.5*(lmbda/2+mu)*ln(Jc) + 0.5*mu*(inner(F,F)-3) # Neo-Hookean type energy
    S = (0.5*lmbda*Jc*invC - (0.5*lmbda+mu)*invC + mu*I) #Could make it (1-d_n)**2 to make it fully decoupled
    P = F*S    
elif energy_type == 'svk':
    #Note that the energy has Sun's paper form and the Stress has Saint-Venant's form
    tr_eps = tr(E_gr)
    tr_ep_sq = tr(E_gr*E_gr)

    psi_e = (1./2)*lmbda*(tr_eps**2) + mu*tr_ep_sq

    psi_e_pos = (1./2)*lmbda*(mac_pos(E_gr1+E_gr2)**2) + mu*(mac_pos(E_gr1)**2 + mac_pos(E_gr2)**2)
    psi_e_neg = (1./2)*lmbda*(mac_neg(E_gr1+E_gr2)**2) + mu*(mac_neg(E_gr1)**2 + mac_neg(E_gr2)**2)

    # Saint Venant Kirchoff
    S = lmbda*tr_eps*I + 2*mu*E_gr
    P = F*S    

else:
    print(f'Invalid energy type, {energy_type}, check parameter file.')
    exit()

cauchy_stress = (1/JF)*F*P.T # Note that this is non-degraded stress (check with Liz)


# Initialize parameters
coefAlpha = []
coefBeta = []
coefZeta = []

for i in range(num_alpha):
    coefAlpha.append(Constant(0.0))
coefAlpha[0].assign(1.0)
for i in range(num_beta):
    coefBeta.append(Constant(1.0))
for i in range(num_zeta):
    coefZeta.append(Constant(0.0))


# Initialize basis coefficients
exactAlpha = []
exactBeta = []
exactZeta = []

alphaSum = 0
for i,key in enumerate(degradation_basis.keys()):
    try:
        alphaValue = degradation_type[key]
        exactAlpha.append(Constant(alphaValue))
        alphaSum += alphaValue
    except:
        print(f"Key '{key}' not found in degradation type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        exactAlpha.append(Constant(0.0))
if alphaSum > 1:
    print('ERROR: Sum of alpha values cannot be greater than 1')
    exit()
exactBeta.append(Constant(eta/(gc*f_c))) #this includes the viscocity and constant
for i,key in enumerate(geometric_basis.keys()):
    try:
        betaValue = geometric_type[key]
        exactBeta.append(Constant(betaValue)*pfbeta)
    except:
        print(f"Key '{key}' not found in geometric type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        exactBeta.append(Constant(0.0))
for i,key in enumerate(history_basis.keys()):
    try:
        zetaValue = dsq_type[key]
        exactZeta.append(Constant(zetaValue)*pfzeta)
    except:
        print(f"Key '{key}' not found in dsquiggle type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        exactZeta.append(Constant(0.0))


g_d = 0
g_d_prime = 0
for count,key in enumerate(degradation_basis.keys()):
    g_d += exactAlpha[count]*degradation_basis[key]['func']()
    g_d_prime += exactAlpha[count]*degradation_basis[key]['dfunc']()
# Note that beta[0] is the coefficient of d(damage)/dt part
f_d = 0
f_d_prime = 0
for count,key in enumerate(geometric_basis.keys()):
    f_d += exactBeta[count+1]*geometric_basis[key]['func']()
    f_d_prime += exactBeta[count+1]*geometric_basis[key]['dfunc']()

print('Functions and Variables setup, defining objective and gradient functions')

sys.stdout.flush() # HPC Only: Force print to the output file


# -------------------------------------------------------------------
#            Pre-Load States and History into Arrays
# -------------------------------------------------------------------
print(f'\nLoading states and history variables from {num_t_steps} steps')


def getLinearMatrices(save_arr,first_pass,last_pass):
    # Initialize empty arrays for assignment
    init_time = time.time()
    dt_arr = []
    state_arr = []
    Htemp_arr = []
    num_save = len(save_arr)
    state_arr_global_idx = np.zeros((num_save))
    for stepIdx in range(num_save):
        state_arr.append(Function(V))
        tmp = []
        for i in range(num_zeta):
            tmp.append(Function(V_h))
        Htemp_arr.append(tmp)

    # print('      Empty arrays populated', flush=True) # Leave for debugging any random job exits

    gd_lin = 0
    R1lin = inner((gd_lin + eps_val)*P, grad(w_u))*dx_
    temp1=assemble(R1lin) # Assemble over shape functions
    temp1 = temp1[interior_dofs] # Gets as a regular vector (instead of Dolfin)
    num_shape_fun = temp1.shape[0]

    print(f'      Identified {num_shape_fun} shape functions') # Leave for debugging any random job exits

    # History still needs to be built up from initial state, so this starts at 0
    for zeta in range(num_zeta):
        H_temp[zeta].assign(zero)
    state_arr_idx = 1
    # load_step_arr = 0,10,20,...
    for stepIdx in range(1,len(load_step_arr)):
        if LOAD_COARSE:
            step_path = os.path.join(load_path,f'steps_{coarse_mesh_idx}')
        else:
            step_path = os.path.join(load_path,'steps')
        # The indices loaded should be global, from the original time series
        curr_step_idx = load_step_arr[stepIdx]
        prev_step_idx = load_step_arr[stepIdx-1]
        # Load the current and previous state
        curr_file_name = os.path.join(step_path,f'result_{curr_step_idx}.h5')
        prev_file_name = os.path.join(step_path,f'result_{prev_step_idx}.h5')
        state_frame_name = '/state'

        if os.path.exists(curr_file_name) == False:
            print(f'Could not find {curr_file_name}, exiting...')
            exit()
        with HDF5File(MPI.comm_world, curr_file_name, 'r') as file_hdf5:
            try:
                # print(file_hdf5)
                file_hdf5.read(state,state_frame_name)
                t_curr = load_curve[curr_step_idx,TINDEX]
            except:
                print(f'ERROR: Could not read step {curr_step_idx} from individual step file, exiting...')
                print(f'       File path: {curr_file_name}')
                exit()
        if os.path.exists(prev_file_name) == False:
            print(f'Could not find {prev_file_name}, exiting...')
        with HDF5File(MPI.comm_world, prev_file_name, 'r') as file_hdf5:
            try:
                file_hdf5.read(state_n,state_frame_name)
                t_prev = load_curve[prev_step_idx,TINDEX]
            except:
                print(f'ERROR: Could not read step {prev_step_idx} from individual step file, exiting...')
                print(f'       File path: {prev_file_name}')
                exit()

        if ADD_NOISE:
            if THRESHOLD_NOISE:
                noise_thresh = 2*noise_sigma
                
            state_len = len(state.sub(0).sub(0).vector()[interior_dofs])
            random_seed = np.random.randint(5000)

            np.random.seed(random_seed)
            noise_vec = np.random.normal(noise_mu, noise_sigma, state_len)

            if THRESHOLD_NOISE:
                noise_idx = abs(noise_vec)<=noise_thresh
                noise_vec[noise_idx==True] = 0

            if noise_field == 'ux':
                state.sub(0).sub(0).vector()[interior_dofs] += noise_vec
            elif noise_field == 'uy':
                state.sub(0).sub(1).vector()[interior_dofs] += noise_vec
            elif noise_field == 'd':
                state.sub(1).vector()[interior_dofs] += noise_vec
            
            # If greater than 1, make sure previous data has the same noise added to it
            if stepIdx == 1:
                print(f'     Adding noise from a Gaussian distribution with mean={noise_mu:1.3e}, sigma={noise_sigma:1.3e}')
                if THRESHOLD_NOISE:
                    print(f'     Threshold value for noise: {noise_thresh:1.3e}')
                np.random.seed(np.random.randint(5000))
            elif stepIdx > 1:
                np.random.seed(prev_random_seed)
            
            noise_vec = np.random.normal(noise_mu, noise_sigma, state_len)
            if THRESHOLD_NOISE:
                noise_idx = abs(noise_vec)<=noise_thresh
                noise_vec[noise_idx==True] = 0
            if noise_field == 'ux':
                state_n.sub(0).sub(0).vector()[interior_dofs] += noise_vec
            elif noise_field == 'uy':
                state_n.sub(0).sub(1).vector()[interior_dofs] += noise_vec
            elif noise_field == 'd':
                state_n.sub(1).vector()[interior_dofs] += noise_vec
            prev_random_seed = random_seed
        dt = t_curr-t_prev
        tau.assign(dt)

        # Evaluate each history term in the basis for the current state
        for i,key in enumerate(history_basis.keys()):
            H_n[i].assign(project(H_temp[i], V_h, solver_type='cg'))            
            H_temp[i].assign(project(hist(history_basis[key]['func'](), H_n[i]), V_h, solver_type='cg'))
        
        # Only update the arrays of states if it is at the time in the simulation you would like to start from
        # first_pass is because array saving only works every 1000 time steps and indexing should be different after 1000
        if curr_step_idx in save_arr:
            
            # print(f'Saving step {load_step_arr[stepIdx]} to save_arr, index {stepIdx}')
            dt_arr.append(dt)
            # If it is the first pass, the 0th element of the array is the initial step from state_n
            #   while in subsequent passes (i.e. time step 1000 - 2000), the 0th element is the current step
            if first_pass == True:
                state_idx = state_arr_idx
            else:
                state_idx = state_arr_idx-1

            if stepIdx == 1 and first_pass == True:
                print('      Saving state_n to beggining of state_arr')
                state_arr[0] = state_n.copy(deepcopy=True)
                state_arr_global_idx[0] = prev_step_idx
                for i in range(num_zeta):
                    Htemp_arr[0][i] = H_n[i].copy(deepcopy=True)

            state_arr[state_idx] = state.copy(deepcopy=True)
            state_arr_global_idx[state_idx] = curr_step_idx
            for i in range(num_zeta):
                Htemp_arr[state_idx][i] = H_temp[i].copy(deepcopy=True)

            state_arr_idx += 1

    data_load_time = time.time()
    print(f'      Took {data_load_time-init_time:1.3f} s to load all data')
    sys.stdout.flush() # HPC Only: Force print to the output file

    A1mat = np.zeros(((num_save-1)*num_shape_fun,num_alpha))
    A2mat = np.zeros(((num_save-1)*num_shape_fun,num_beta+num_zeta))
    Bvec = np.zeros(((num_save-1)*num_shape_fun))

    sys.stdout.flush() # HPC Only: Force print to the output file

    # Rows of matrix are time steps
    # Start at 1 so previous time step at start can be loaded
    testingData = pd.DataFrame(columns=['step','d','d_diff','hist'])
    for stepIdx in range(1,num_save):
        dt = dt_arr[stepIdx-1]
        tau.assign(dt)

        state.assign(state_arr[stepIdx])
        state_n.assign(state_arr[stepIdx-1])
        for i in range(num_zeta):
            H_temp[i].assign(Htemp_arr[stepIdx][i])
        
        astart = (stepIdx-1)*num_shape_fun
        aend = astart+num_shape_fun
        # Columns of matrix are alphas multipliers
        for count,key in enumerate(degradation_basis.keys()):
            gd_lin = degradation_basis[key]['func']()

            R1lin = inner((gd_lin + eps_val)*P, grad(w_u))*dx_ 

            # tem1 will be the residual for all shape functions, it will be a vector
            tem1=assemble(R1lin) # Assemble over shape functions
            tem1 = tem1[interior_dofs] # Gets as a regular vector (instead of Dolfin)

            A1mat[astart:aend,count] = tem1

        #### Second residual/matrix for second inference

        # First Column
        save_damage = assemble(inner(d, d)*dx_) # COMMENT LATER, debugging only
        save_damage_diff = assemble(inner((d-d_n)/tau, (d-d_n)/tau)*dx_) # COMMENT LATER, debugging only
        
        R2col1 = (1/tau)*inner(d-d_n, w_d)*dx_
        
        tempVec=assemble(R2col1) # Assemble over shape functions
        tempVec = tempVec[interior_dofs] # Gets as a regular vector
        
        A2mat[astart:aend,0] = tempVec

        # Beta Columns
        for count,key in enumerate(geometric_basis.keys()):
            fd_prime_lin = geometric_basis[key]['dfunc']()
            R2colbeta = inner(w_d,fd_prime_lin)*dx_
            
            tempVec=assemble(R2colbeta) # Assemble over shape functions
            tempVec = tempVec[interior_dofs] # Gets as a regular vector
            
            A2mat[astart:aend,count+1] = tempVec

        # Zeta Columns
        save_his = assemble(inner(H_temp[0], H_temp[0])*dx_) # COMMENT LATER, debugging only
        for count,key in enumerate(history_basis.keys()):
            # For now, use exact g_d_prime using known alpha
            R2colzeta = inner(w_d, g_d_prime)*H_temp[count]*dx_ 
            
            tempVec=assemble(R2colzeta) # Assemble over shape functions
            tempVec = tempVec[interior_dofs] # Gets as a regular vector
            
            A2mat[astart:aend,count+num_beta] = tempVec 

        ### The b vector for Ax-b
        R2b =  inner(grad(w_d), grad(d))*dx_
        tempVec=assemble(R2b) # Assemble over shape functions
        tempVec = tempVec[interior_dofs] # Gets as a regular vector

        Bvec[astart:aend] = tempVec

        testingData = pd.concat([pd.DataFrame([[save_arr[stepIdx],save_damage,save_damage_diff,save_his]], columns=testingData.columns), testingData], ignore_index=True)

    sys.stdout.flush() # HPC Only: Force print to the output file

    del state_arr, Htemp_arr
    del tempVec, R2b, R2colzeta, R2colbeta, R2col1, tem1, R1lin
    sys.stdout.flush() # HPC Only: Force print to the output file

    print(f'      Global indices: {state_arr_global_idx[0:10]}...{state_arr_global_idx[-10:num_save]}')
    print(f'      A1: {A1mat.shape}, A2: {A2mat.shape}, B: {Bvec.shape}')
    return A1mat,A2mat,Bvec,testingData


sys.stdout.flush() # HPC Only: Force print to the output file

num_steps_arr = len(step_save_arr)
ARRAY_CUTOFF = 1000
if num_steps_arr < ARRAY_CUTOFF:
    print('Less than 1000 steps, generating in one pass')
    A1mat,A2mat,Bvec,testingData = getLinearMatrices(step_save_arr,True,False)
else:
    print('  More than 1000 steps, iteratively generating matrices', flush=True)
    sub_start = 0
    sub_end = ARRAY_CUTOFF
    first_pass_bool = True
    last_pass_bool = False
    print(f'  Iterations ending when start index > {step_save_arr[-1]}', flush=True)
    while sub_start < step_save_arr[-1]:
        sub_step_arr = step_save_arr[sub_start:sub_end]
        if len(sub_step_arr) == 1:
            sub_step_arr = step_save_arr[sub_start-1:sub_end]
            last_pass_bool = True
        elif len(sub_step_arr) == 0:
            print(f'sub_step_arr is of 0 length, sub_start = {sub_start}, step_save_arr = {step_save_arr[-1]}')
        print(f'\n    Generating for {sub_step_arr[0]} to {sub_step_arr[-1]}, ({len(sub_step_arr)} steps)', flush=True)
        A1tmp,A2tmp,Btmp,testingDF = getLinearMatrices(sub_step_arr,first_pass_bool,last_pass_bool)
        if sub_start == 0:
            A1mat = A1tmp
            A2mat = A2tmp
            Bvec = Btmp
            testingData = testingDF
            del A1tmp, A2tmp, Btmp
        else:
            A1mat = np.vstack((A1mat,A1tmp))
            del A1tmp
            A2mat = np.vstack((A2mat,A2tmp))
            del A2tmp
            Bvec = np.hstack((Bvec,Btmp))
            del Btmp
            testingData = pd.concat([testingDF, testingData], ignore_index=True)

        sub_start += ARRAY_CUTOFF
        sub_end += ARRAY_CUTOFF
        first_pass_bool = False
        if sub_end > step_save_arr[-1]:
            sub_end = step_save_arr[-1]

print(f'\nDone loading data, shapes are:\n  A1: {A1mat.shape}, A2: {A2mat.shape}, B: {Bvec.shape}', flush=True)


A1_save_path = os.path.join(save_path,f'A1_{save_name}.npy')
testDF_path = os.path.join(save_path,f'testingDF_{save_name}.pickle')
print(f'Saving the matrix to the file: {A1_save_path}', flush=True)
# Seg Fault with 32 cores, trying with 48...but on compute...still seg faulted
A1 = np.array(A1mat)
np.save(A1_save_path,A1)
print(f'Saving the testing data to pickle file: {testDF_path}', flush=True)
testingData.to_pickle(testDF_path)

A2_save_path = os.path.join(save_path,f'A2_{save_name}.npy')
print(f'Saving the matrix to the file: {A2_save_path}', flush=True)
A2 = np.array(A2mat)
np.save(A2_save_path,A2)

B_save_path = os.path.join(save_path,f'B_{save_name}.npy')
print(f'Saving the matrix to the file: {B_save_path}', flush=True)
Bsave = np.array(Bvec)
np.save(B_save_path,Bsave)

print('Exiting after saving array.')