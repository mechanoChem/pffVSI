# --------------------------------
# VSI: Variational System Identification
# PFF: Phase Field Fracture
# Viscoelastic crack growth with nonlinear elasticity
# 
# Author: Siddhartha Srivastava (sidsriva@umich.edu) and Liz Livingston (lizliv@umich.edu)
# Date: Aug 29, 2023
# Updated: Sep 1, 2025
# See PFF2D_tensile.py for forward solve

# Usage:
#     python3 VSI_PFF2D_nonlinear.py simulations/simXX/params.json
# --------------------------------
from ufl import *
from fenics import *
import numpy as np
import scipy.optimize as sp
import os
import sys
import argparse
import json
import shutil

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
    g_d         = pow(one-d,2)
    return g_d
def degradation_deriv_quadratic():
    g_d_prime   = -2*(one-d)
    return g_d_prime

def degradation_quasi_quadratic():
    p=10 #Fixing m and p based on Sun's paper
    m=50 
    g_d = ((1.-d)**2)/((1.-d)**2 + m*d*(1.+p*d))
    return g_d
def degradation_deriv_quasi_quadratic():
    p=10 #Fixing m and p based on Sun's paper
    m=50 
    g_d_prime = ((d-1.)*(d*(2.*p+1.) + 1.)*m)/(((d**2)*(m*p+1.) + d*(m-2.) + 1.)**2)
    return g_d_prime

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
    # return psi_e
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
    # M1, M2 = pc.projectionTensors(P, lam1, lam2) #Use this in future if the projection tensors are needed to align with fiber directions etc.
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
#                   VSI SETTINGS
# ------------------------------------------------
DTINDEX = 2
TINDEX = 1

ADD_NOISE = True
noise_type = 'ux'
noise_mu, noise_sigma = 0, 5e-8 # mean and standard deviation

OVERWRITE_START_END = True # Set to true if you want to set new_start_index and new_num_t_steps
new_start_index = 739
new_num_t_steps = 1046-739

SCALING_COEF = 1
MULTIPLY_DT = False
DIVIDE_NUM_STEPS = True

SUBSAMPLE_TIME = True
subsample_size = 2

OVERWRITE_QUAD_DEGREE = False
new_quad_degree = 20

print(f'Adding noise to {noise_type} with mean={noise_mu},sd={noise_sigma}\n')
# #------------------------------------------------
# #          Define Parameters
# #------------------------------------------------
parser = argparse.ArgumentParser(description='Assign simulation parameters and select solver type.')
parser.add_argument('paramFile', metavar='N', type=str,
                    help='name or path to parameter file')
args = parser.parse_args()
param_file = args.paramFile

print('Current Directory:',os.getcwd())

load_path,param_file = setupSimulationFiles(param_file)
print(f'     Load path is: {load_path}')
print(f'     Parameter file is: {param_file}\n')
param_path = os.path.join(load_path,param_file)

# This will load the parameters and then dynamically generate variables from the keys of the json file
with open(param_path, "r") as read_file:
    params = json.load(read_file)

# Mesh/Geom information
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

# Get the array of steps that need to be loaded (starts from 0)
if SUBSAMPLE_TIME == False:
    load_step_arr = np.arange(0,end_index+1)
elif SUBSAMPLE_TIME == True:
    load_step_arr = np.arange(0,end_index+1,subsample_size)

# Get the array of steps to be saved/used in inference (can be greater than 0)
step_idx = np.where(load_step_arr>=start_index)
step_save_arr = load_step_arr[step_idx]

start_index = step_save_arr[0]
end_index = step_save_arr[-1]
num_t_steps = len(step_save_arr)
if SUBSAMPLE_TIME == True:
    print(f'     Inferring every {subsample_size} steps from {start_index} to {end_index} ({num_t_steps} time steps)\n')
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

# Check that the mesh actually exists
mesh_path = f'./mesh_files/{mesh_name}'
if os.path.exists(mesh_path) == False:
    print(f'ERROR: Mesh does not exist, please check file name/path in parameter file: {mesh_path}')
    exit()
# Process the mesh file extension
mesh_type = mesh_name.split('.')
if len(mesh_type) < 2:
    print('ERROR: Did not include file extension in mesh_name, please update parameter file')
    exit()
else:
    mesh_type = mesh_type[1]
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
u_temp, d_temp  = split(state_temp) # u_n.split(True)for postprocessing

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

#Estimate boundary dofs
temp_func = Function(V)
temp_func.vector()[:]=2
temp_bc = DirichletBC(V.sub(0).sub(0), zero, "on_boundary" )
temp_bc.apply(temp_func.vector())
interior_dofs = np.flatnonzero(temp_func.vector().get_local() == 2).astype(np.int32)
boundary_dofs = np.flatnonzero(temp_func.vector().get_local() == 0).astype(np.int32)
ndof = len(interior_dofs)

#x-(horizontal), y-(vertical) 
top     = CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.max())
bottom  = CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.min())
left    = CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.min())
right   = CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.max())

#Get BC for current time step
curr_top_disp    = Expression("t", t = 0.0, degree=1)

BC_u_top            = DirichletBC(V.sub(0).sub(1), curr_top_disp, top)
BC_u_bottom_full    = DirichletBC(V.sub(0), Constant((0.0,0.0)), bottom)
BC_u_bottom         = DirichletBC(V.sub(0).sub(1), zero, bottom)
BC_u_right          = DirichletBC(V.sub(0).sub(0), zero, right)
BC_u_left_corner    = DirichletBC(V.sub(0).sub(0), zero, "x[0]<0.0+DOLFIN_EPS && x[1]<0.0+DOLFIN_EPS", method="pointwise")

if bc_cond == 'bottom_full':
    bcs = [BC_u_top, BC_u_bottom_full]
elif bc_cond == 'bottom_roller':
    # bcs = [BC_u_top, BC_u_bottom]
    bcs = [BC_u_top, BC_u_bottom,BC_u_left_corner]

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
    g_d += coefAlpha[count]*degradation_basis[key]['func']()
    g_d_prime += coefAlpha[count]*degradation_basis[key]['dfunc']()
# Note that beta[0] is the coefficient of d(damage)/dt part
f_d = 0
f_d_prime = 0
for count,key in enumerate(geometric_basis.keys()):
    f_d += coefBeta[count+1]*geometric_basis[key]['func']()
    f_d_prime += coefBeta[count+1]*geometric_basis[key]['dfunc']()

# Deformation residual
R1 = inner((g_d + eps_val)*P, grad(w_u))*dx_  
# Damage residue
# Weak form: phase-field equation
R2 = coefBeta[0]*(1/tau)*inner(d-d_n, w_d)*dx_
R2 +=  inner(grad(w_d), grad(d))*dx_
R2 += inner(w_d,f_d_prime)*dx_ 
for count,key in enumerate(history_basis.keys()):
    R2 += inner(w_d, g_d_prime)*coefZeta[count]*H_temp[count]*dx_

# Define gradient
dR1dtheta=[0]*num_alpha
dR2dtheta=[0]*(num_beta + num_zeta)

for i in range(num_alpha):
  dR1dtheta[i]=diff(R1,coefAlpha[i])
for i in range(num_beta):
  dR2dtheta[i]=diff(R2,coefBeta[i])
for i in range(num_zeta):
  dR2dtheta[num_beta + i]=diff(R2,coefZeta[i])



# -------------------------------------------------------------------
#            Pre-Load States and History into Arrays
# -------------------------------------------------------------------
print(f'\nLoading states and history variables from {num_t_steps} steps')
# Initialize empty arrays for assignment
init_time = time.time()
dt_arr = []
state_arr = []
Htemp_arr = []
for stepIdx in range(len(step_save_arr)+1):
    state_arr.append(Function(V))
    tmp = []
    for i in range(num_zeta):
        tmp.append(Function(V_h))
    Htemp_arr.append(tmp)

print('      Empty arrays populated', flush=True)

# History still needs to be built up from initial state, so this starts at 0
for zeta in range(num_zeta):
    H_temp[zeta].assign(zero)
state_arr_idx = 1
for stepIdx in range(1,len(load_step_arr)):
    # The indices loaded should be global, from the original time series
    curr_step_idx = load_step_arr[stepIdx]
    prev_step_idx = load_step_arr[stepIdx-1]
    # Load the current and previous state
    file_name = os.path.join(load_path,'steps',f'result_{curr_step_idx}.h5')
    state_frame_name = '/state'
    with HDF5File(MPI.comm_world, file_name, 'r') as file_hdf5:
        try:
            file_hdf5.read(state,state_frame_name)
            t_curr = load_curve[curr_step_idx,TINDEX]
        except:
            print(f'Could not read step {curr_step_idx} from individual step file, exiting...')
            exit()
    file_name = os.path.join(load_path,'steps',f'result_{prev_step_idx}.h5')
    with HDF5File(MPI.comm_world, file_name, 'r') as file_hdf5:
        try:
            file_hdf5.read(state_n,state_frame_name)
            t_prev = load_curve[prev_step_idx,TINDEX]
        except:
            print(f'Could not read step {prev_step_idx} from individual step file, exiting...')
            exit()

    if ADD_NOISE:
        state_len = len(state.sub(0).sub(0).vector()[interior_dofs])
        random_seed = np.random.randint(5000)
        np.random.seed(random_seed)
        if noise_type == 'ux':
            state.sub(0).sub(0).vector()[interior_dofs] += np.random.normal(noise_mu, noise_sigma, state_len)
        elif noise_type == 'uy':
            state.sub(0).sub(1).vector()[interior_dofs] += np.random.normal(noise_mu, noise_sigma, state_len)
        elif noise_type == 'd':
            state.sub(1).vector()[interior_dofs] += np.random.normal(noise_mu, noise_sigma, state_len)
        # If greater than 1, make sure previous data has the same noise added to it
        if stepIdx == 1:
            print(f'     Adding noise from a Gaussian distribution with mean={noise_mu}, sigma={noise_sigma}')
            np.random.seed(np.random.randint(5000))
        elif stepIdx > 1:
            np.random.seed(prev_random_seed)
        # Now update the previous state with appropriate noise
        if noise_type == 'ux':
            state_n.sub(0).sub(0).vector()[interior_dofs] += np.random.normal(noise_mu, noise_sigma, state_len)
        elif noise_type == 'uy':
            state_n.sub(0).sub(1).vector()[interior_dofs] += np.random.normal(noise_mu, noise_sigma, state_len)
        elif noise_type == 'd':
            state_n.sub(1).vector()[interior_dofs] += np.random.normal(noise_mu, noise_sigma, state_len)
        prev_random_seed = random_seed
    dt = t_curr-t_prev
    tau.assign(dt)


    # Evaluate each history term in the basis for the current state
    for i,key in enumerate(history_basis.keys()):
        H_n[i].assign(project(H_temp[i], V_h, solver_type='cg'))            
        H_temp[i].assign(project(hist(history_basis[key]['func'](), H_n[i]), V_h, solver_type='cg'))
    
    # Only update the arrays of states if it is at the time in the simulation you would like to start from
    if curr_step_idx in step_save_arr:
        if state_arr_idx > len(state_arr):
            print(f'ERROR: state_arr_index {state_arr_idx} is larger than len(state_array) {len(state_arr)}',flush=True)
        dt_arr.append(dt)
        if curr_step_idx == start_index:
            state_arr[0] = state_n.copy(deepcopy=True)
            for i in range(num_zeta):
                Htemp_arr[0][i] = H_n[i].copy(deepcopy=True)
        state_arr[state_arr_idx] = state.copy(deepcopy=True)
        for i in range(num_zeta):
            Htemp_arr[state_arr_idx][i] = H_temp[i].copy(deepcopy=True)
        state_arr_idx += 1

data_load_time = time.time()
print(f'     Took {data_load_time-init_time:1.3f} s to load all data')

#Loss and Gradient of losses functions
# 1 for inferring alpha (stress equilibrium residual)
# 2 for inferring beta and zeta (damage evolution)

print('\nLoss function effects:')
print(f'  Multiplying loss functions by dt: {MULTIPLY_DT}')
print(f'  Divide loss by num_steps: {DIVIDE_NUM_STEPS}')
print(f'  Scaling the loss/gradients by: {SCALING_COEF}')

def objective_fun1(x):
    #update weights
    for i in range(num_alpha):
        coefAlpha[i].assign(x[i])  
    
    loss1=0
    
    for stepIdx in range(len(step_save_arr)):
        dt = dt_arr[stepIdx]
        tau.assign(dt)

        state.assign(state_arr[stepIdx])

        tem1=assemble(R1)
        tem1 = tem1[interior_dofs] # vector of residual evaluated for all test functions
        if MULTIPLY_DT == True:
            loss1 += dt*np.inner(tem1,tem1)
        else:
            loss1 += np.inner(tem1,tem1)

    if DIVIDE_NUM_STEPS == True:
        return SCALING_COEF*(1/num_t_steps)*loss1
    else:
        return SCALING_COEF*loss1
def gradient_fun1(x):
    #update weights
    for i in range(num_alpha):
        coefAlpha[i].assign(x[i])  
    
    dl1dtheta=np.zeros(num_alpha)

    for stepIdx in range(len(step_save_arr)):
        dt = dt_arr[stepIdx]
        tau.assign(dt)

        state.assign(state_arr[stepIdx])

        R1_ = assemble(R1)
        R1_ = R1_[interior_dofs]
        for i in range(num_alpha):
            dR1dtheta_i = assemble(dR1dtheta[i])
            dR1dtheta_i = dR1dtheta_i[interior_dofs]
            tem = np.dot(dR1dtheta_i,R1_)
            if MULTIPLY_DT == True:
                dl1dtheta[i] += dt*2*tem
            else:
                dl1dtheta[i] += 2*tem

    if DIVIDE_NUM_STEPS == True:
        return SCALING_COEF*(1/num_t_steps)*dl1dtheta
    else:
        return SCALING_COEF*dl1dtheta

def objective_fun2(x, alpha_vals):
    #update weights
    for i in range(num_alpha):
        coefAlpha[i].assign(alpha_vals[i])  
    for i in range(num_beta):
        coefBeta[i].assign(x[i])  
    for i in range(num_zeta):
        coefZeta[i].assign(x[i + num_beta])
    loss2=0

    for stepIdx in range(1,len(step_save_arr)):
        dt = dt_arr[stepIdx]
        tau.assign(dt)

        state.assign(state_arr[stepIdx])
        state_n.assign(state_arr[stepIdx-1])
        
        for i in range(num_zeta):
            H_temp[i].assign(Htemp_arr[stepIdx][i])
    
        tem2=assemble(R2)
        tem2 = tem2[interior_dofs]
        if MULTIPLY_DT == True:
            loss2 += dt*np.inner(tem2,tem2)
        else:
            loss2 += np.inner(tem2,tem2)
    if DIVIDE_NUM_STEPS == True:
        return SCALING_COEF*(1/num_t_steps)*loss2
    else:
        return SCALING_COEF*loss2
def gradient_fun2(x, alpha_vals):
    #update weights
    for i in range(num_alpha):
        coefAlpha[i].assign(alpha_vals[i])  
    for i in range(num_beta):
        coefBeta[i].assign(x[i])  
    for i in range(num_zeta):
        coefZeta[i].assign(x[i + num_beta])
    
    dl2dtheta=np.zeros(num_beta + num_zeta)

    for stepIdx in range(1,len(step_save_arr)):
        dt = dt_arr[stepIdx]
        tau.assign(dt)
        
        state.assign(state_arr[stepIdx])
        state_n.assign(state_arr[stepIdx-1])
        
        for i in range(num_zeta):
            H_temp[i].assign(Htemp_arr[stepIdx][i])
    
        R2_=assemble(R2)
        R2_ = R2_[interior_dofs]
        for i in range(num_beta):
            dR2dtheta_i=assemble(dR2dtheta[i])
            dR2dtheta_i = dR2dtheta_i[interior_dofs]
            tem=np.dot(dR2dtheta_i,R2_)
            if MULTIPLY_DT == True:
                dl2dtheta[i] += dt*2*tem
            else:
                dl2dtheta[i] += 2*tem
        for i in range(num_zeta):
            dR2dtheta_i=assemble(dR2dtheta[i+num_beta])
            dR2dtheta_i = dR2dtheta_i[interior_dofs]
            tem=np.dot(dR2dtheta_i,R2_)
            if MULTIPLY_DT == True:
                dl2dtheta[num_beta+i] += dt*2*tem
            else:
                dl2dtheta[num_beta+i] += 2*tem
                
    if DIVIDE_NUM_STEPS == True:
        return SCALING_COEF*(1/num_t_steps)*dl2dtheta
    else:
        return SCALING_COEF*dl2dtheta

def iter_cb(m):
    print ("m = ", m)

def iter_cb2_1(m):
    print ("     m = ", m, ", loss =", objective_fun1(m))

def iter_cb2_2(m):
    print ("     m = ", m, ", loss =", objective_fun2(m,x_alpha))

def eval_cb(j, m):
    print ("j = %f, m = %f." % (j, m))

def derivative_cb(j, dj, m):
    print("j = %f, dj = %s, m = %s." %(j, [float(k) for k in dj], [float(k) for k in m]))


if __name__ == "__main__":

    sys.stdout.flush() # HPC Only: Force print to the output file

    start_time = time.time()
    
    x_alpha = [0.5,0.5]
    alpha_starts = [[0.2,0.8],[0.8,0.2]]
    x_beta_zeta = np.random.rand(num_beta+num_zeta)*np.random.randint(1,3)

    # If not given, chosen method will be one of BFGS, L-BFGS-B, SLSQP (gave nans on docker), depending on whether or not the problem has constraints or bounds
    # Unconstrained methods:  CG, BFGS, Newton-CG
    # Bounded methods: L-BFGS-B (seg faults on docker), TNC (x wasnt updating on docker)

    # The first method must take bounds and constraints
    optimization_method_1 = 'SLSQP'
    # opt_method_list_1 = ['trust-constr'] # 'COBYLA'
    # The second method needs to take bounds
    optimization_method_2 = 'TNC'
    opt_method_list_2 = ['L-BFGS-B'] #,'Nelder-Mead','Powell'] # Apparently these do not use the gradient
    print(f'\nOptimizing using: {optimization_method_1} for problem 1 and {optimization_method_2} for problem 2')

    ########################################
    #        First Objective Function
    ########################################

    # Positive alphas
    bounds1=np.zeros((num_alpha,2))
    bounds1[:,1]=1.
    # Sum of alphas is equal to 1
    cons1 = {'type':'eq', 'fun': lambda x: np.sum(x)-1}

    print('\n-----------------------------------------------\nObjective Function 1\n-----------------------------------------------')

    init_res_loss = objective_fun1(x_alpha)
    init_res_grad = gradient_fun1(x_alpha)
    
    # Check loss and gradient, if gradient is > e+3, scale enough to reduce it down
    gradient_limit = 1e3
    large_grad = init_res_grad > gradient_limit
    if True in large_grad:
        print('\n  ERR: Gradient is too large\n  Optimization likely would not converge, decreasing scaling coefficient')
        print(f'     gradient = {init_res_grad}')
        o1_start = time.time()
        max_reductions = 10
        reduction_idx = 0
        while True in large_grad and reduction_idx < max_reductions:
            SCALING_COEF = SCALING_COEF/10
            init_res_grad = gradient_fun1(x_alpha)
            large_grad = init_res_grad > gradient_limit
            reduction_idx += 1
        o1_end = time.time()
        print(f'    gradient = {init_res_grad}, time: {o1_end-o1_start:1.3f} s')
        print(f'    New scaling coefficient: {SCALING_COEF:1.2e}')

    sys.stdout.flush() # HPC Only: Force print to the output file
    
    o1_start = time.time()
    init_res_loss = objective_fun1(exactAlpha)
    init_res_grad = gradient_fun1(exactAlpha)
    o1_end = time.time()
    print(f'  Using the exact coefficients:')
    print(f'    loss: {init_res_loss}, gradient: {init_res_grad}')
    print(f'    Time to compute: {o1_end-o1_start:1.3f} s')

    sys.stdout.flush() # HPC Only: Force print to the output file

    init_res_loss = objective_fun1(x_alpha)
    init_res_grad = gradient_fun1(x_alpha)
    print('\n  Minimizing first objective function:')
    print(f'     m_0 = {x_alpha}, loss = {init_res_loss}, gradient = {init_res_grad}')

    # print('\n--------------------------------------')
    o1_start = time.time()
    obj1 = sp.minimize(objective_fun1, x_alpha,jac=gradient_fun1,method=optimization_method_1, constraints=cons1, bounds=bounds1, callback=iter_cb2_1)
    o1_end = time.time()
    # print('--------------------------------------')
    print(f'\n  Miminization Time: {o1_end-o1_start:1.3e} s\n')
    print('  SciPy summary:\n',obj1)
    print('--------------------------------------\n')

    sys.stdout.flush() # HPC Only: Force print to the output file

    obj1_success = obj1.success
    obj1_alphas = obj1.x
    if obj1_success == False:
        print(f'Failed to converge with {optimization_method_1}\n')
    if np.sum(obj1_alphas)-1 > 1e-2:
        print('Constraint not satisfied...')
    if np.linalg.norm(obj1_alphas-x_alpha) == 0.0:
        print('Alpha was not updated...')

    # Loop to try different Alpha starting points if minimization failed, constraint isn't satisfied, or alphas did not change
    obj1_meth_idx = 0
    while obj1_success == False or np.sum(obj1_alphas)-1 > 1e-2 or np.linalg.norm(obj1_alphas-x_alpha) == 0.0 or obj1_meth_idx > len(alpha_starts):
        print(f'\nAttempting to use alternative start: {alpha_starts[obj1_meth_idx]}')
        x_alpha = alpha_starts[obj1_meth_idx]
        init_res_loss = objective_fun1(x_alpha)
        init_res_grad = gradient_fun1(x_alpha)
        print(f'     m_0 = {x_alpha}, loss = {init_res_loss}, gradient = {init_res_grad}')
        o1_start = time.time()
        obj1 = sp.minimize(objective_fun1, x_alpha,jac=gradient_fun1,method=optimization_method_1, constraints=cons1, bounds=bounds1, callback=iter_cb2_1)
        o1_end = time.time()
        print(f'\n  Miminization Time: {o1_end-o1_start:1.3e} s\n')
        print('  SciPy summary:\n',obj1)
        obj1_success = obj1.success
        obj1_alphas = obj1.x

        obj1_meth_idx += 1


    print('\n------------------------------------------------------------')
    # print('Updating alpha based on first objective function...')
    print('Updating alpha based on exact for second objective function...')
    
    print('\nInference Summary for Alpha Coefficients')
    print(f'    Initial Guess: {x_alpha[0]:1.3e}, {x_alpha[1]:1.3e}')
    for i in range(len(x_alpha)):
        # x_alpha[i] = obj1.x[i]
        x_alpha[i] = float(exactAlpha[i])
    print(f'    Discovered:    {obj1.x[0]:1.3e}, {obj1.x[1]:1.3e}')
    print(f'    Exact:         {float(exactAlpha[0]):1.3e}, {float(exactAlpha[1]):1.3e}')
    print('------------------------------------------------------------')
    
    sys.stdout.flush() # HPC Only: Force print to the output file

    # Reset the scaling coefficient and check the value of the second residual
    SCALING_COEF = 1

    ########################################
    #        Second Objective Function
    ########################################

    bounds2=np.zeros((num_beta + num_zeta,2)) #Positive geometric and driving force
    bounds2[:,1]= np.inf


    print('\n-----------------------------------------------\nObjective Function 2\n-----------------------------------------------')

    o2_start = time.time()
    init_res_loss = objective_fun2(exactBeta+exactZeta, exactAlpha)
    init_res_grad = gradient_fun2(exactBeta+exactZeta, exactAlpha)
    o2_end = time.time()
    print(f'  Using the exact coefficients:')
    print(f'    loss: {init_res_loss}, gradient: {init_res_grad}')
    print(f'    Time to compute: {o2_end-o2_start:1.3f} s')
    
    sys.stdout.flush() # HPC Only: Force print to the output file

    init_res_loss = objective_fun2(x_beta_zeta, x_alpha)
    init_res_grad = gradient_fun2(x_beta_zeta, x_alpha)
    
    print('\n  Minimizing second objective function:')
    print(f'     m_0 = {x_beta_zeta}, loss = {init_res_loss}, gradient = {init_res_grad}')

    o2_start = time.time()
    obj2 = sp.minimize(objective_fun2, x_beta_zeta,jac=gradient_fun2,method=optimization_method_2, args=(x_alpha), bounds=bounds2, callback=iter_cb2_2)
    o2_end = time.time()
    
    print(f'\n  Miminization Time: {o2_end-o2_start:1.3e} s\n')
    print('  SciPy summary:\n',obj2)
    print('--------------------------------------\n')

    print(f'                            beta_0,      linear,      single,      double')
    print(f'    Initial Guess:         {x_beta_zeta[0]:1.3e},   {x_beta_zeta[1]:1.3e},   {x_beta_zeta[2]:1.3e},   {x_beta_zeta[3]:1.3e}')
    print(f'    Discovered (from SI):  {obj2.x[0]:1.3e},   {obj2.x[1]:1.3e},   {obj2.x[2]:1.3e},   {obj2.x[3]:1.3e}')
    print(f'    Exact (from Forward):  {float(exactBeta[0]):1.3e},   {float(exactBeta[1]):1.3e},   {float(exactBeta[2]):1.3e},   {float(exactBeta[3]):1.3e}')
    
    print('                            energy')
    print(f'    Initial Guess:         {x_beta_zeta[num_beta+0]:1.3e}')
    print(f'    Discovered (from SI):  {obj2.x[num_beta+0]:1.3e}')
    print(f'    Exact (from Forward):  {float(exactZeta[0]):1.3e}')

    obj2_success = obj2.success
    obj2_betazetas = obj2.x
    if obj2_success == False:
        print(f'Failed to converge with {optimization_method_2}\n')
    if np.linalg.norm(obj2_betazetas-x_beta_zeta) == 0.0:
        print('Beta/Zetas not updated...')


    end_time = time.time()
    total_sim_time = end_time-start_time
    if total_sim_time <= 60:
        print(f'\n\nMain File Took {total_sim_time:1.3e} s')
    elif total_sim_time > 60 and total_sim_time <= 3600:
        print(f'\n\nMain File Took {total_sim_time:1.3e} s = {total_sim_time/60:1.2f} min')
    else:
        print(f'\n\nMain File Took {total_sim_time:1.3e} s = {total_sim_time/3600:1.2f} hr')

    sys.stdout.flush() # HPC Only: Force print to the output file



    # Loop to try different optimization method
    obj2_meth_idx = 0
    print(f'\n\nAttempting to use alternative solver: {opt_method_list_2[obj2_meth_idx]}')
    optimization_method_2 = opt_method_list_2[obj2_meth_idx]
    init_res_loss = objective_fun2(x_beta_zeta, x_alpha)
    init_res_grad = gradient_fun2(x_beta_zeta, x_alpha)
    print(f'     m_0 = {x_beta_zeta}, loss = {init_res_loss}, gradient = {init_res_grad}')
    o2_start = time.time()
    obj2 = sp.minimize(objective_fun2, x_beta_zeta,jac=gradient_fun2,method=optimization_method_2, args=(x_alpha), bounds=bounds2, callback=iter_cb2_2)
    o2_end = time.time()
    print(f'\n  Miminization Time: {o2_end-o2_start:1.3e} s\n')
    print('  SciPy summary:\n',obj2)
    obj2_success = obj2.success
    obj2_betazetas = obj2.x

    print('\nInference Summary for Beta Coefficients')
    if num_beta == 4:
        print(f'                            beta_0,      linear,      single,      double')
        print(f'    Initial Guess:         {x_beta_zeta[0]:1.3e},   {x_beta_zeta[1]:1.3e},   {x_beta_zeta[2]:1.3e},   {x_beta_zeta[3]:1.3e}')
        print(f'    Discovered (from SI):  {obj2.x[0]:1.3e},   {obj2.x[1]:1.3e},   {obj2.x[2]:1.3e},   {obj2.x[3]:1.3e}')
        print(f'    Exact (from Forward):  {float(exactBeta[0]):1.3e},   {float(exactBeta[1]):1.3e},   {float(exactBeta[2]):1.3e},   {float(exactBeta[3]):1.3e}')
    if num_beta == 3:
        # print(f'                            beta_0,      linear,      single')
        print(f'                            beta_0,      linear,      double')
        print(f'    Initial Guess:         {x_beta_zeta[0]:1.3e},   {x_beta_zeta[1]:1.3e},   {x_beta_zeta[2]:1.3e}')
        print(f'    Discovered (from SI):  {obj2.x[0]:1.3e},   {obj2.x[1]:1.3e},   {obj2.x[2]:1.3e}')
        print(f'    Exact (from Forward):  {float(exactBeta[0]):1.3e},   {float(exactBeta[1]):1.3e},   {float(exactBeta[2]):1.3e}')

    print('\nInference Summary for Zeta Coefficients')
    if num_zeta == 2:
        print('                            energy,     principle')
        print(f'    Initial Guess:         {x_beta_zeta[num_beta+0]:1.3e},   {x_beta_zeta[num_beta+1]:1.3e}')
        print(f'    Discovered (from SI):  {obj2.x[num_beta+0]:1.3e},   {obj2.x[num_beta+1]:1.3e}')
        print(f'    Exact (from Forward):  {float(exactZeta[0]):1.3e},   {float(exactZeta[1]):1.3e}')
    if num_zeta == 1:
        print('                            energy')
        print(f'    Initial Guess:         {x_beta_zeta[num_beta+0]:1.3e}')
        print(f'    Discovered (from SI):  {obj2.x[num_beta+0]:1.3e}')
        print(f'    Exact (from Forward):  {float(exactZeta[0]):1.3e}')

    
    end_time = time.time()
    total_sim_time = end_time-start_time
    if total_sim_time <= 60:
        print(f'\n\nMain File Took {total_sim_time:1.3e} s')
    elif total_sim_time > 60 and total_sim_time <= 3600:
        print(f'\n\nMain File Took {total_sim_time:1.3e} s = {total_sim_time/60:1.2f} min')
    else:
        print(f'\n\nMain File Took {total_sim_time:1.3e} s = {total_sim_time/3600:1.2f} hr')
