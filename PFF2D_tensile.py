# --------------------------------
# PFF: Phase Field Fracture
# Viscoelastic crack growth with nonlinear elasticity
# 
# Author: Siddhartha Srivastava (sidsriva@umich.edu) and Liz Livingston (lizliv@umich.edu)
# Date: March 02, 2023
# Updated: Sep 1, 2025
#
# Based on: DOI 10.1007/s10237-015-0702-0
#-------------------------------------
from ufl import *
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json
import shutil

import principal_components as pc
from postprocessing.combine_results import plot_load_curve

set_log_active(True) # Dolfins logging setup

import logging
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)

parameters['linear_algebra_backend'] = 'Eigen'

import time
time_start = time.time()


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
    # Only the tensile component of the energy should drive damage
    if energy_type == 'svk':
        return psi_e_pos
    else:
        return psi_e
def dsq_energyCrit():
    # For now this needs psi critical
    if energy_type == 'svk':
        psi_i = psi_e_pos - psi_crit
    else:
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
def hist(dsq,h_old):
    return conditional(ge(dsq, h_old), dsq, h_old)
def H(H_old):
  
  psi_i_new = psi_e - psi_crit
  psi_i_old = psi_e_n - psi_crit
  
  psi_new =  psi_crit + (1./2.)*(psi_i_new + abs(psi_i_new))
  psi_old =  psi_crit + (1./2.)*(psi_i_old + abs(psi_i_old))

  return conditional(lt(psi_old, psi_new), psi_new, H_old)

degradation_basis = {'quad': {'name': 'quadratic', 'func':degradation_quadratic, 'dfunc': degradation_deriv_quadratic}, 
                     'quasiQuad': {'name': 'quasiQuadratic', 'func':degradation_quasi_quadratic, 'dfunc': degradation_deriv_quasi_quadratic},
                     }
geometric_basis = {'linear': {'name': 'linear', 'func':geometric_linear, 'dfunc': geometric_deriv_linear}, 
                     'singleWell': {'name': 'singleWell', 'func':geometric_single_well, 'dfunc': geometric_deriv_single_well},
                     'doubleWell': {'name': 'doubleWell', 'func':geometric_double_well, 'dfunc': geometric_deriv_double_well},
                     }
history_basis = {'energy': {'name': 'energy', 'func':dsq_energy, },
                'energyCrit': {'name': 'energyCrit', 'func':dsq_energyCrit,},
                'principal': {'name': 'principal', 'func':dsq_principal,},
                'tensile': {'name': 'tensile', 'func':dsq_sigma22,} ,
                }


#------------------------------------------------
#          Define Parameters
#------------------------------------------------
parser = argparse.ArgumentParser(description='Assign simulation parameters and select solver type.')
parser.add_argument('paramFile', metavar='N', type=str,
                    help='name or path to parameter file')
args = parser.parse_args()
param_file = args.paramFile

load_path,param_file = setupSimulationFiles(param_file)
print(f'     Load path is: {load_path}')
print(f'     Parameter file is: {param_file}')
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
        
#------------------------------------------------
#          Initialize basis coefficients
#------------------------------------------------
num_alpha = len(degradation_basis.keys())
num_beta = len(geometric_basis.keys())+1 #+1 for the gradient energy term
num_zeta = len(history_basis.keys())

pfzeta = Constant(1/(gc*f_c))
coefAlpha = []
coefBeta = []
coefBeta.append(Constant(eta/(gc*f_c))) #this includes the viscocity and constant
coefZeta = []

alphaSum = 0
for i,key in enumerate(degradation_basis.keys()):
    try:
        alphaValue = degradation_type[key]
        coefAlpha.append(Constant(alphaValue))
        alphaSum += alphaValue
    except:
        print(f"Key '{key}' not found in degradation type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        coefAlpha.append(Constant(0.0))
if alphaSum > 1:
    print('ERROR: Sum of alpha values cannot be greater than 1')
    exit()
for i,key in enumerate(geometric_basis.keys()):
    try:
        betaValue = geometric_type[key]
        coefBeta.append(Constant(betaValue)*pfbeta)
    except:
        print(f"Key '{key}' not found in geometric type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        coefBeta.append(Constant(0.0))
for i,key in enumerate(history_basis.keys()):
    try:
        zetaValue = dsq_type[key]
        coefZeta.append(Constant(zetaValue)*pfzeta)
    except:
        print(f"Key '{key}' not found in dsquiggle type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        coefZeta.append(Constant(0.0))


print('\nBased on the parameter file, we have:')
print(f'Alpha = {[float(item) for item in coefAlpha]}')
print(f'Beta = {[float(item) for item in coefBeta]}')
print(f'Zeta = {[float(item) for item in coefZeta]}')


#-------------------------------------------------
#           Basic Loading/printout
#-------------------------------------------------
load_curve = np.zeros((100,5)) # Keep this array small for memory. Append as needed
dt_index = 2
start_index = 0
start_time = 0
if os.path.exists(os.path.join(load_path,'load_curve.txt')) == False:
    print(f"\nNo restart files ({os.path.join(load_path,'load_curve.txt')}) exist, cannot read checkpoint\n")
    read_checkpoint = False
if read_checkpoint==True:
    read_curve = np.loadtxt(os.path.join(load_path,'load_curve.txt'), delimiter=' ')
    start_index = len(read_curve[:,0])-1
    start_time = read_curve[start_index,1]
    last_dt = read_curve[-1,dt_index]
    print(f'\nUpdating time step to {last_dt} based on previous run')
    dt = last_dt
    tau = Constant(dt)
    load_curve = np.vstack((read_curve,load_curve))
    print(f"Loaded data from previous run, ended at step number: {start_index}, t={start_time}\n")
end_index = start_index+num_t_steps

# ---------------------------------------------------------------- 
# Read mesh
# ----------------------------------------------------------------
# Define mesh
from dolfin import *

# Process the mesh file extension
mesh_type = mesh_name.split('.')
if len(mesh_type) < 2:
    print('ERROR: Did not include file extension in mesh_name, please update parameter file',flush=True)
    exit()
else:
    mesh_type = mesh_type[1]

# Check that the mesh actually exists
mesh_path = f'./mesh_files/{mesh_name}'
if os.path.exists(mesh_path) == False:
    print(f'ERROR: Mesh does not exist, please check file name/path in parameter file: {mesh_path}',flush=True)
    exit()

# Load the Mesh
if mesh_type == 'xml':
    print('Loading xml file',flush=True)
    mesh = Mesh('./mesh_files/'+mesh_name)
elif mesh_type == 'xdmf':
    print('Loading xdmf file',flush=True)
    mesh = Mesh()
    with XDMFFile(f"./mesh_files/{mesh_name}") as file:
        file.read(mesh)

dx_ = dx(metadata={'quadrature_degree': q_degree})
x = SpatialCoordinate(mesh)
ndim = mesh.topology().dim()

time_load = time.time()
print(f'Basic loading in: {time_load-time_start:1.3e} s\n',flush=True)

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
state       = Function(V)   # The current time step
state_n     = Function(V)   # The previous time step
state_temp  = Function(V)   # For implicit integration scheme

# Split system functions to access components
u, d            = split(state)      # u.split(True) for postprocessing 
u_n, d_n        = split(state_n)    # u_n.split(True)for postprocessing
u_temp, d_temp  = split(state_temp) # u_n.split(True)for postprocessing

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

dof_coordinates = V.tabulate_dof_coordinates()
print('DOF coordinate shape', dof_coordinates.shape)
dof_coordinates.resize((V.dim(), mesh.geometry().dim()))
print(f'Dimensions: V_u {V.dim()}, Mesh {mesh.geometry().dim()}')
dof_x = dof_coordinates[:, 0] 
dof_y = dof_coordinates[:, 1]
spec_height = dof_y.max()-dof_y.min()
print(f'Geometry dimensions\n  x={dof_x.min(),dof_x.max()}, y={dof_y.min(),dof_y.max()}',flush=True)
print(f'The specimen height is: {spec_height} mm')


# Debugging: Print num elements and vertices
c = Circumradius(mesh)
c_vector=project(c,FunctionSpace(mesh, "DG", 0)).vector()[:] * 2
print(f"Element Sizes: Min={min(c_vector):1.3f}\n  Max={max(c_vector):1.3f}, Avg={np.average(c_vector):1.3f}")

elements = mesh.cells()
edges = mesh.num_edges()
num_vertices = mesh.num_vertices()
print(f'Mesh Size\n    {len(elements)} elements, {edges}, edges, {num_vertices} vertices')

#x-(horizontal), y-(vertical) 
top     = CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.max())
bottom  = CompiledSubDomain("near(x[1], side) && on_boundary", side = dof_y.min())
left    = CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.min())
right   = CompiledSubDomain("near(x[0], side) && on_boundary", side = dof_x.max())

#Get BC for current time step
curr_top_disp    = Expression("t", t = 0.0, degree=1)
curr_bot_disp    = Expression("t", t = 0.0, degree=1)

BC_u_top            = DirichletBC(V.sub(0).sub(1), curr_top_disp, top)
BC_u_bottom_disp    = DirichletBC(V.sub(0).sub(1), curr_bot_disp, bottom)
BC_u_bottom_full    = DirichletBC(V.sub(0), Constant((0.0,0.0)), bottom)
BC_u_bottom         = DirichletBC(V.sub(0).sub(1), zero, bottom)
BC_u_right          = DirichletBC(V.sub(0).sub(0), zero, right)
BC_u_left_corner    = DirichletBC(V.sub(0).sub(0), zero, "x[0]<0.0+DOLFIN_EPS && x[1]<0.0+DOLFIN_EPS", method="pointwise")

if bc_cond == 'bottom_full':
    bcs = [BC_u_top, BC_u_bottom_full]
elif bc_cond == 'bottom_roller':
    bcs = [BC_u_top, BC_u_bottom, BC_u_left_corner]
elif bc_cond == 'top_bottom_disp':
    bcs = [BC_u_top, BC_u_bottom_disp, BC_u_left_corner]

# Mark the top edge for traction/load evaluation
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
top.mark(boundary_subdomains,1)

#Estimate boundary dofs
temp_func = Function(V_u)
temp_func.vector()[:]=2
[bc.apply(temp_func.vector()) for bc in bcs]
interior_dofs = np.flatnonzero(temp_func.vector().get_local() == 2).astype(np.int32)
temp_func = Function(V_h)
temp_func.vector()[:]=2
[bc.apply(temp_func.vector()) for bc in bcs]
damage_interior_dofs = np.flatnonzero(temp_func.vector().get_local() == 2).astype(np.int32)

print('Boundary conditions defined',flush=True)
#------------------------------------------------
#            Load in state/history
#------------------------------------------------

H_n = []
H_temp = []
for count,key in enumerate(history_basis.keys()):
    H_n.append(Function(V_h)) # Need a scalar field for storing history
    H_temp.append(Function(V_h)) # Need a scalar field for storing history of previous time step
# Initial Condition = Start with undeformed virgin material
state.assign(zero_state)

# Update the state based on previous results
if read_checkpoint==True:
    # If loading a previous simulation, load the state and history
    with HDF5File(MPI.comm_world, os.path.join(load_path,'result.h5'), 'r') as file_hdf5:
        frame_name = '/state_%s'%(str(start_index))
        file_hdf5.read(state,frame_name)
        for i in range(num_zeta):
            frame_name = f'/history_{start_index}-zeta_{i}'
            file_hdf5.read(H_n[i],frame_name)

    print('Read in checkpoint file', start_index)
else:
    # Otherwise, set everything to zero and save initial results
    for i in range(num_zeta):
        H_n[i].assign(zero)
    with HDF5File(MPI.comm_world, os.path.join(load_path,'result.h5'), 'w') as file_hdf5: 
        file_hdf5.write(mesh,'/mesh')
        frame_name = '/state_%s'%(str(0))
        file_hdf5.write(state,frame_name)
        for i in range(num_zeta):
            frame_name = f'/history_{0}-zeta_{i}'
            file_hdf5.write(H_n[i],frame_name)
    with HDF5File(MPI.comm_world, os.path.join(load_path,'steps',f'result_{0}.h5'), 'w') as file_hdf5: 
        file_hdf5.write(mesh,'/mesh')
        file_hdf5.write(state,'/state')
        for i in range(num_zeta):
            frame_name = f'/history-zeta_{i}'
            file_hdf5.write(H_n[i],frame_name)


#------------------------------------------------
#            Constitutive Relationships
#------------------------------------------------
            
#Elasticity 
I = Identity(ndim)          # Identity Tensor
F = I + grad(u)             # Deformation Gradient
F_n = I + grad(u_n)         # Deformation Gradient

FiT = transpose(inv(F))
Ce = F.T*F                # Right Cauchy-Green Tensor
E_gr = 0.5*(Ce-I)        # Right Cauchy-Green Strain Tensor
E_gr_n = 0.5*(F_n.T*F_n - I)

E_gr1,E_gr2 = pc.eigenValues(E_gr,tol=1e-15)

def mac_pos(A):
    return (abs(A)+A)/2
def mac_neg(A):
    return (abs(A)-A)/2

# Invariants of Deformation Tensors
I1      = tr(Ce)
I2      = 0.5*(I1*I1-tr(Ce*Ce) )
I3      = det(Ce)

J       = det(F)
Jc      = det(Ce)
invC    = inv(Ce)

barI1   = (J**(-2/3))*I1

beta    = 2*nu/(1 - 2*nu)
# Linear elastic relationships
lmbda   = (2*mu*nu)/(1 - 2*nu)
kappa   = lmbda + (2*mu)/3
print(f'With linear elastic relationships, mu={mu:1.3f} and nu={nu:1.3f} result in:')
print(f'    lambda={lmbda:1.3f} and kappa={kappa:1.3f}', flush=True)

# 4th invariants for Anisotropic tensor
a1 = Constant((np.cos(np.deg2rad(gamma)), -np.sin(np.deg2rad(gamma))))
a2 = Constant((np.cos(np.deg2rad(gamma)), np.sin(np.deg2rad(gamma))))
M1 = outer(a1,a1)
M2 = outer(a2,a2)
CM1 = tr(Ce.T*M1)
CM2 = tr(Ce.T*M2)
Fa1a1 = outer(dot(F,a1), a1)
Fa2a2 = outer(dot(F,a2), a2)

# Energy
if energy_type == 'svk':
    #Note that the energy has Sun's paper form and the Stress has Saint-Venant's form
    tr_eps = tr(E_gr)
    tr_ep_sq = tr(E_gr*E_gr)
    psi_e = (1./2)*lmbda*(tr_eps**2) + mu*tr_ep_sq

    psi_e1 = (1./2)*lmbda*((E_gr1+E_gr2)**2) + mu*(E_gr1*E_gr1 + E_gr2*E_gr2)

    psi_e_pos = (1./2)*lmbda*(mac_pos(E_gr1+E_gr2)**2) + mu*(mac_pos(E_gr1)**2 + mac_pos(E_gr2)**2)
    psi_e_neg = (1./2)*lmbda*(mac_neg(E_gr1+E_gr2)**2) + mu*(mac_neg(E_gr1)**2 + mac_neg(E_gr2)**2)

    psi_e2 = psi_e_pos + psi_e_neg

    # NOTE: Required for Suh's history function
    tr_eps_n = tr(E_gr_n)
    tr_ep_sq_n = tr(E_gr_n*E_gr_n)
    psi_e_n = (1./2)*lmbda*(tr_eps_n**2) + mu*tr_ep_sq_n

    # Saint Venant Kirchoff
    S = lmbda*tr(E_gr)*I + 2*mu*E_gr
    P = F*S
elif energy_type == 'cnh':
    # Compressible neo-hookean model
    psi_e = 0.25*lmbda*(Jc-1) - 0.5*(lmbda/2+mu)*ln(Jc) + 0.5*mu*(I1-2) 
    S = (0.5*lmbda*Jc*invC - (0.5*lmbda+mu)*invC + mu*I) 
    P = F*S
elif energy_type == 'ninh':
    # Nearly incompressible neo-hookean model
    psi_e = 0.5*kappa*(ln(J)**2) + 0.5*mu*(barI1-2)
    S = kappa*ln(J)*invC + mu*(J**(-2/3))*(-(1/3)*I1*invC + I)
    P = F*S
elif energy_type == 'hgo':
    ## First we need a definition for fiber directions
    UVW = Constant((np.cos(np.deg2rad(gamma)), -np.sin(np.deg2rad(gamma))))
    
    C_10 = 1.05e3
    kappa = 7.44e4
    k1 = 2.95e4
    k2 = 2.14
    theta_6 = 9.76e-3
    
    I4 = dot(UVW,Ce*UVW)
    
    invFT=inv(F).T
    B=F*F.T
    Fa=F*UVW
    I14=theta_6*I1+(1-3*theta_6)*I4-1
    
    P = C_10*J**(-2./3.)*(F-1./3.*I1*invFT)
    P+= kappa*J*(J-1)*invFT
    P+= k1*I14*exp(k2*I14*I14)*(theta_6*B+(1-3*theta_6)*outer(Fa, Fa))*invFT

    psi_e  = 0.5*C_10*(J**(-2./3.)*I1 - 3)
    psi_e += 0.5*kappa*(J-1)**2
    psi_e += 0.5*(k1/k2)*exp(k2*I14*I14)

    # Still need positive and negative to form the basis, won't be used with stress based criteria
    psi_e_pos = (1./2)*lmbda*(mac_pos(E_gr1+E_gr2)**2) + mu*(mac_pos(E_gr1)**2 + mac_pos(E_gr2)**2)
    psi_e_neg = (1./2)*lmbda*(mac_neg(E_gr1+E_gr2)**2) + mu*(mac_neg(E_gr1)**2 + mac_neg(E_gr2)**2)
elif energy_type == 'anisotropic':
    psi_e = (mu/beta)*(pow(J,-beta)-1) + 0.5*mu*(I1-2) # Raina 2016 (Eq 36)
    # Anisotropic energy components, Raina 2016 (Eq 37)
    if k2 != 0:
        psi_e += (k1/k2) * ( exp(k2*(CM1 - 1)) - 1) - k1*(CM1 - 1)
        psi_e += (k1/k2) * ( exp(k2*(CM2 - 1)) - 1) - k1*(CM2 - 1)

    # Still need positive and negative to form the basis, won't be used with Raina's stress based criteria
    psi_e_pos = (1./2)*lmbda*(mac_pos(E_gr1+E_gr2)**2) + mu*(mac_pos(E_gr1)**2 + mac_pos(E_gr2)**2)
    psi_e_neg = (1./2)*lmbda*(mac_neg(E_gr1+E_gr2)**2) + mu*(mac_neg(E_gr1)**2 + mac_neg(E_gr2)**2)
    
    P = -mu*pow(J,-beta)*FiT + mu*F       #Nonlinear elastic stress without damage
    P += 2*k1*(exp(k2*(CM1 - 1))-1)*Fa1a1  #Anisotropy
    P += 2*k1*(exp(k2*(CM2 - 1))-1)*Fa2a2
else:
    print(f'Energy type {energy_type} not found, check parameter file.')
    exit()

# cauchy_stress = cauchy()
cauchy_stress = (1/J)*F*P.T

# Calculation of tensile load for load/displacement curve
traction_top_surface = dot(P, unit_y_normal)
ds_top = Measure("ds", domain=mesh, subdomain_data=boundary_subdomains, subdomain_id=1, metadata={'quadrature_degree': q_degree})
tensile_load = dot(traction_top_surface , unit_y_normal) *ds_top


#------------------------------------------------
#            Residual Setup
#------------------------------------------------
g_d = 0
g_d_prime = 0
f_d = 0
f_d_prime = 0
# D_squiggle = 0

for count,key in enumerate(degradation_basis.keys()):
    g_d += coefAlpha[count]*degradation_basis[key]['func']()
    g_d_prime += coefAlpha[count]*degradation_basis[key]['dfunc']()

#Note that beta[0] is the coefficient of d(damage)/dt part
for count,key in enumerate(geometric_basis.keys()):
    f_d += coefBeta[count+1]*geometric_basis[key]['func']()
    f_d_prime += coefBeta[count+1]*geometric_basis[key]['dfunc']()

# NOTE: Check monolithic generic residual formulation
if update_type == "monolithic": 
    # Damage residue
    R = coefBeta[0]*(1/tau)*inner(d-d_n, w_d)*dx_
    R +=  inner(grad(w_d), grad(d))*dx_
    R += inner(w_d,f_d_prime)*dx_ 
    for count,key in enumerate(history_basis.keys()):
        R += coefZeta[count]*inner(w_d, g_d_prime)*H_n[count]*dx_
    # Deformation residue
    R += inner((g_d + eps_val)*P, grad(w_u))*dx_ 

    J = derivative(R, state)
elif update_type == "explicit" or update_type == "implicit":
    # Displacement residual
    R1 = inner((g_d + eps_val)*P, grad(w_u))*dx_  
    R1+= inner(d-d_temp, w_d)*dx_  #Weak form for d=d_n

    # Damage residual
    R2 = coefBeta[0]*(1/tau)*inner(d-d_n, w_d)*dx_
    R2 +=  inner(grad(w_d), grad(d))*dx_
    R2 += inner(w_d,f_d_prime)*dx_ 
    for count,key in enumerate(history_basis.keys()):
        if update_type == "explicit":
            R2 += coefZeta[count]*inner(w_d, g_d_prime)*H_n[count]*dx_
        elif update_type == "implicit":
            R2 += coefZeta[count]*inner(w_d, g_d_prime)*H_temp[count]*dx_
    R2 +=  inner(u-u_temp, w_u)*dx_ #Weak form for u=u_n
    
    J1 = derivative(R1, state)
    J2 = derivative(R2, state)
else: 
    print('Invalid update type') 



time_setup = time.time()
print(f'Setup FeniCs functions and variables in: {time_setup-time_load:1.3e} s')

sys.stdout.flush() # HPC Only: Force print to the output file

state_n.assign(state)
curr_t = start_time + dt
tau.assign(dt)
exceeded = False
t_index = start_index+1
converge_index = t_index
time_reduction_count = 0
time_reduction_limit = 5
while t_index <= end_index:
    time_step_start = time.time()
    print('\n----------------------------------------------------------------', flush=True)
    print(f'                      t={curr_t:1.4f} , step={t_index-start_index}/{num_t_steps}', flush=True)
    print('----------------------------------------------------------------', flush=True)

    # Update time step and load
    curr_top_disp.t = params['disp_rate']*curr_t
    curr_bot_disp.t = -params['disp_rate']*curr_t

    try:
        if update_type == "monolithic": 
            problem = NonlinearVariationalProblem(R, state, bcs, J)
            solver = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm['newton_solver']['linear_solver'] = 'gmres'
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-9
            prm['newton_solver']['maximum_iterations'] = 20
            solver.solve()
            temp = project(conditional(ge(D_sqiggle, H_n), D_sqiggle, H_n), V_h, solver_type='cg')
            H_n.assign(temp)
            state_n.assign(state)
            solver_diverged = False
        elif update_type == "explicit" or update_type == "implicit":
            
            sub_iter_count  = 0
            temp_error = 1
            
            # Update the temporary variables for staggered scheme
            for idx,key in enumerate(history_basis.keys()):
                H_temp[idx].assign(H_n[idx])
            state_temp.assign(state_n)
            
            while temp_error > err_tol and sub_iter_count < max_subiter:
                sub_iter_count+=1

                #Solve for displacement field
                problem1 = NonlinearVariationalProblem(R1, state, bcs, J1)
                solver1 = NonlinearVariationalSolver(problem1)
                prm = solver1.parameters
                prm['newton_solver']['absolute_tolerance'] = 1E-6
                prm['newton_solver']['relative_tolerance'] = 1E-6
                prm['newton_solver']['maximum_iterations'] = 10
                prm['newton_solver']['relaxation_parameter'] = 0.8
                
                print('                                    Solving for displacements', flush=True)
                print('----------------------------------------------------------------------------------------------------', flush=True)
                solver1.solve()
                temp_error_u = assemble(inner(u - u_temp, u - u_temp)*dx_)

                # Update temporary history/state variables (Compare history against the previos step)
                for idx,key in enumerate(history_basis.keys()):
                    if hist_fun_type == "cpg":
                        H_temp[idx].assign(project(hist(history_basis[key]['func'](), H_n[idx]), V_h, solver_type='cg'))
                    elif hist_fun_type == "suh":
                        H_temp[idx].assign(project(H(history_basis[key]['func'](), H_n[idx]), V_h, solver_type='cg'))
                    else:
                        print('Invalid history function type, check parameter file.', flush=True)
                        exit()
                state_temp.assign(state) #Update displacement field
                    
                #Solve for damage field
                problem2 = NonlinearVariationalProblem(R2, state, bcs, J2)
                solver2 = NonlinearVariationalSolver(problem2)
                prm = solver2.parameters        
                prm['newton_solver']['absolute_tolerance'] = 1E-6
                prm['newton_solver']['relative_tolerance'] = 1E-6
                prm['newton_solver']['maximum_iterations'] = 10
                
                print('\n                                    Solving for damage', flush=True)
                print('----------------------------------------------------------------------------------------------------', flush=True)
                solver2.solve()                      
                
                temp_error_d = assemble(inner(d - d_temp, d - d_temp)*dx_)
                temp_error = np.sqrt(temp_error_u + temp_error_d)
                
                print(f'subiter = {sub_iter_count}, fixed point iteration error = {temp_error}', flush=True)
                
                # Update temporary state (damage field only, history is displacement which R2 doesn't change)
                state_temp.assign(state) 

            # Update previous history/state solutions
            state_n.assign(state)
            for idx,key in enumerate(history_basis.keys()):
                H_n[idx].assign(project(H_temp[idx], V_h, solver_type='cg'))
             
            solver_diverged = False
            if sub_iter_count >= max_subiter:
                exceeded= True
                print(f'    Exceeded {max_subiter} implicit steps, error was:', flush=True)
                print(f'       {temp_error} where tolerance was {err_tol}', flush=True)

            print('----------------------------------------------------------------------------------------------------', flush=True)
        else:
            print('Invalid update type', flush=True)
            break 

    except Exception as e: 
        solver_diverged = True
        if t_index == 1:
            converge_index = start_index
        print(f'Solver diverged at t index={t_index}', flush=True)
        time_diverged = time.time()
        print(f'Solver ran for: {time_diverged-time_step_start:1.3e} s', flush=True)
        print(e)
        break
    
    # If the solver has not divered, update time, save results, etc.
    if solver_diverged == False and exceeded==False:
        converge_index = t_index

        # Timing and housekeeping
        time_converge = time.time()
        phys_t = time_converge-time_step_start

        # strain, stress
        load_curve[converge_index, :] = [phys_t, curr_t, dt, curr_top_disp.t/spec_height, assemble(tensile_load)] #Hard coded strain for current geometry
        print(f'\n     Nominal Strain (%): {load_curve[converge_index, 3]}, Tensile Load (N) : {load_curve[converge_index, 4]}\n', flush=True)   

        with HDF5File(MPI.comm_world, os.path.join(load_path,'result.h5'), 'a') as file_hdf5: 
            frame_name = '/state_%s'%(str(converge_index))
            file_hdf5.write(state,frame_name)
            for i in range(num_zeta):
                frame_name = f'/history_{converge_index}-zeta_{i}'
                file_hdf5.write(H_n[i],frame_name)
        with HDF5File(MPI.comm_world, os.path.join(load_path,'steps',f'result_{converge_index}.h5'), 'w') as file_hdf5: 
            file_hdf5.write(mesh,'/mesh')
            file_hdf5.write(state,'/state')
            for i in range(num_zeta):
                frame_name = f'/history-zeta_{i}'
                file_hdf5.write(H_n[i],frame_name)

        # Add another 100 spaces to the load curve as necessary
        if t_index%100 > 0:
            load_curve = np.vstack((load_curve,np.zeros((100,5))))
        
        np.savetxt(os.path.join(load_path,'load_curve.txt'), load_curve[0:converge_index+1,:], fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#, encoding=None)

        curr_t += dt

        print(f'Time step took: {phys_t:1.3e} s\n\n', flush=True)
        t_index +=1
    
    # If the subiterations are exceeded, reduced the time step and try again
    if exceeded==True and t_index>1 and time_reduction_count < 5:
        #re-load previous state info to ensure solve didn't overwrite it
        with HDF5File(MPI.comm_world, os.path.join(load_path+'/result.h5'), 'r') as file_hdf5:
            frame_name = '/state_%s'%(str(t_index-1))
            file_hdf5.read(state,frame_name)
            for i in range(num_zeta):
                frame_name = f'/history_{t_index-1}-zeta_{i}'
                file_hdf5.read(H_n[i],frame_name)

        #load previous time step, reduce time step 
        dt=dt/2
        curr_t += dt
        time_reduction_count += 1
        exceeded=False
        print(f'Did not achieve tolerance reducing time step to {dt:1.3e}', flush=True)

    if exceeded==True and t_index==1:
        print('Did not achieve tolerance on first time step') 
        break
    if exceeded==True and time_reduction_count==time_reduction_limit:
        print('Did not achieve tolerance even after reducing time step many times') 
        break
        

# End of time loop, save results and figure
if converge_index > 2:
    np.savetxt(os.path.join(load_path,'load_curve.txt'), load_curve[0:converge_index+1,:], fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#, encoding=None)

    plot_load_curve(load_curve[0:converge_index+1,:],load_path)

time_done = time.time()
print(f'{num_t_steps} steps took: {time_done-time_setup:1.3e} s')
