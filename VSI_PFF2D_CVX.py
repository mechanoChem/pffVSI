# --------------------------------
# VSI: Variational System Identification
# PFF: Phase Field Fracture
# Script to solve linear minimization problem for system inference
#
# Author: Liz Livingston
# Date: September 2025
#
# Usage:
#     python VSI_PFF2D_CVX.py simulations/simXX/params.json
# --------------------------------
import os
import sys
import time
import argparse
import json
import numpy as np
import cvxpy as cp
import scipy.optimize as sp
import pandas as pd
import csv 

import logging

program_start = time.time()

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


def print_xalpha(alphaCoef):
    # Print the solution
    logger.info(f'    x={alphaCoef}')

    logger.info('\n  Inference Summary for Alpha Coefficients')
    logger.info(f'    Discovered:    {alphaCoef[0]:1.3e}, {alphaCoef[1]:1.3e}')
    logger.info(f'    Exact:         {float(exactAlpha[0]):1.3e}, {float(exactAlpha[1]):1.3e}')
    logger.info('------------------------------------------------------------')
def print_xbetazeta(x_beta_zeta):
    # logger.info the solution
    logger.info(f'    x={x_beta_zeta}')
    
    logger.info('\n  Inference Summary for Beta Coefficients')
    if num_beta == 4:
        logger.info(f'                            beta_0,      linear,      single,      double')
        logger.info(f'    Discovered (from SI):  {x_beta_zeta[0]:1.3e},   {x_beta_zeta[1]:1.3e},   {x_beta_zeta[2]:1.3e},   {x_beta_zeta[3]:1.3e}')
        logger.info(f'    Exact (from Forward):  {float(exactBeta[0]):1.3e},   {float(exactBeta[1]):1.3e},   {float(exactBeta[2]):1.3e},   {float(exactBeta[3]):1.3e}')
    if num_beta == 3:
        # logger.info(f'                            beta_0,      linear,      single')
        logger.info(f'                            beta_0,      linear,      double')
        logger.info(f'    Discovered (from SI):  {x_beta_zeta[0]:1.3e},   {x_beta_zeta[1]:1.3e},   {x_beta_zeta[2]:1.3e}')
        logger.info(f'    Exact (from Forward):  {float(exactBeta[0]):1.3e},   {float(exactBeta[1]):1.3e},   {float(exactBeta[2]):1.3e}')
    if num_beta == 2:
        logger.info(f'                            beta_0,      single')
        logger.info(f'    Discovered (from SI):  {x_beta_zeta[0]:1.3e},   {x_beta_zeta[1]:1.3e}')
        logger.info(f'    Exact (from Forward):  {float(exactBeta[0]):1.3e},   {float(exactBeta[1]):1.3e}')

    logger.info('\n  Inference Summary for Zeta Coefficients')
    if num_zeta == 2:
        logger.info('                            energy,     principle')
        logger.info(f'    Discovered (from SI):  {x_beta_zeta[num_beta+0]:1.3e},   {x_beta_zeta[num_beta+1]:1.3e}')
        logger.info(f'    Exact (from Forward):  {float(exactZeta[0]):1.3e},   {float(exactZeta[1]):1.3e}')
    if num_zeta == 1:
        logger.info('                            energy')
        logger.info(f'    Discovered (from SI):  {x_beta_zeta[num_beta+0]:1.3e}')
        logger.info(f'    Exact (from Forward):  {float(exactZeta[0]):1.3e}')

degradation_basis = {
                    'quad': {'name': 'quadratic'}, 
                     'quasiQuad': {'name': 'quasiQuadratic'},
                     }
geometric_basis = {
                    'linear': {'name': 'linear'}, 
                     'singleWell': {'name': 'singleWell'},
                     'doubleWell': {'name': 'doubleWell'},
                     }
history_basis = {'energy': {'name': 'energy'},
                }

num_alpha = len(degradation_basis.keys())
num_beta = len(geometric_basis.keys())+1 #+1 for the gradient energy term
num_zeta = len(history_basis.keys())




# #------------------------------------------------
# #          Define Parameters
# #------------------------------------------------
parser = argparse.ArgumentParser(description='Assign simulation parameters and select solver type.')
parser.add_argument('paramFile', metavar='N', type=str,
                    help='name or path to parameter file')

parser.add_argument('-noise','--load_noise', help="Flag to add noise", required=False, action='store_true')
parser.add_argument('-nf','--noise_field', help="Noise field to be applied: ux, uy, or d", required=False, default="uy")
parser.add_argument('-sig','--noise_sigma', help="Noise standard deviation (sigma) to be applied: i.e. 5e-7", required=False, default=5e-7)
parser.add_argument('-idx', '--noise_idx', help="Index of the VSI dataset to be added, include if multiple", required=False, default=1)

parser.add_argument('-loadsub','--load_subsample', help="Flag to load subsampled data", required=False, action='store_true')
parser.add_argument('-datasub','--data_subsample', help="Flag to subsample data", required=False, action='store_true')
parser.add_argument('-ss', '--sample_size', help="Subsampling to be applied: i.e. 2 or 4", required=False, default=2)

parser.add_argument('-obj','--obj_func')

args = parser.parse_args()
param_file = args.paramFile

print('Current Directory:',os.getcwd())

load_path,param_file = setupSimulationFiles(param_file)
print(f'  Load path is: {load_path}')
print(f'  Parameter file is: {param_file}\n')
param_path = os.path.join(load_path,param_file)

# This will load the parameters and then dynamically generate variables from the keys of the json file
with open(param_path, "r") as read_file:
    params = json.load(read_file)

# Damage Parameters
ell = params['ell'] # mm
gc = params['g_c']
eta = params['eta'] # 1e-5 MN-s/mm2
# Damage Parameters
degradation_type = params['degradation_type']
geometric_type = params['geometric_type']
dsq_type    = params['dsq_type']
pfIndex = params['pfIndex']

# if/else statement determine f_c, coefficient for beta, coefficient for zeta
if pfIndex==1:
    f_c = 2.*ell
    pfbeta = 1/(8.*ell*ell)
elif pfIndex==2:
    f_c = 2.
    pfbeta = 1/(4.)
elif pfIndex==3:
    f_c = ell
    pfbeta = 1/(2.*l*l)
elif pfIndex==4:
    f_c = 2.*l/(np.pi)
    pfbeta = 1/(2*ell*ell)
elif pfIndex==5:
    f_c = 3*ell/(4)
    pfbeta = 1/(2*ell*ell)
elif pfIndex==6:
    f_c = 3*ell/(4*np.sqrt(2))
    pfbeta = 1/(ell*ell)
else:
    print(f'ERROR: pfIndex of {pfIndex} does not exist')
    exit()

pfzeta = 1/(gc*f_c)


# Initialize basis coefficients
exactAlpha = []
exactBeta = []
exactZeta = []

alphaSum = 0
for i,key in enumerate(degradation_basis.keys()):
    try:
        alphaValue = degradation_type[key]
        exactAlpha.append(alphaValue)
        alphaSum += alphaValue
    except:
        print(f"Key '{key}' not found in degradation type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        exactAlpha.append(0.0)
if alphaSum > 1:
    print('ERROR: Sum of alpha values cannot be greater than 1')
    exit()
exactBeta.append(eta/(gc*f_c)) #this includes the viscocity and constant
for i,key in enumerate(geometric_basis.keys()):
    try:
        betaValue = geometric_type[key]
        exactBeta.append(betaValue*pfbeta)
    except:
        print(f"Key '{key}' not found in geometric type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        exactBeta.append(0.0)
for i,key in enumerate(history_basis.keys()):
    try:
        zetaValue = dsq_type[key]
        exactZeta.append(zetaValue*pfzeta)
    except:
        print(f"Key '{key}' not found in dsquiggle type input, check spelling in parameter file.")
        print(f"     Setting '{key}' coefficient to 0.0")
        exactZeta.append(0.0)

LOAD_FULL = True

# Settings that change the way VSI is performed
LOAD_COARSE = True
coarse_mesh_idx = 'M1'
coarse_mesh_name = f'doublenotch_coarsen_{coarse_mesh_idx}.xdmf'

LOAD_NOISE = False
vsiNum = int(args.noise_idx)
noise_field = args.noise_field
mu = 0
sigma = float(args.noise_sigma)
if args.load_noise:
    LOAD_NOISE = True
    print(f'Adding noise on {noise_field} DOF with sigma of {sigma}')
# noise_sigma = 5e-9 # mean and standard deviation

sampleSize = int(args.sample_size)
# Experimental subsampling
LOAD_SUBSAMPLED     = False
if args.load_subsample:
    LOAD_SUBSAMPLED = True
    print(f'Will load subsample time steps every {sampleSize} steps')
# Data subsampling
SUBSAMPLE_REGIONS   = False
if args.data_subsample:
    SUBSAMPLE_REGIONS = True
    print(f'Will subsample the data every {sampleSize} time steps')

obj_func = float(args.obj_func)
if obj_func == 1:
    SOLVE_FIRST_OBJ     = True
    SOLVE_SECOND_OBJ    = False
elif obj_func == 2:
    SOLVE_FIRST_OBJ     = False
    SOLVE_SECOND_OBJ    = True

if LOAD_COARSE and coarse_mesh_idx == 'M0':
    num_shape_func = 193498
elif LOAD_COARSE and coarse_mesh_idx == 'M3':
    num_shape_func = 49163
else:
    num_shape_func      = 217951 #217690 #217994

ARRAY_CUTOFF = 45e6
SUBTRACT_TIME = 200 # Roughly the array cutoff divided by the num_shape_func
USE_LAST = False

print(f'Will cutoff matrices with sizes larger than {ARRAY_CUTOFF}')
if USE_LAST:
    print(f'   Using the LAST {SUBTRACT_TIME} steps for this cutoff')
else:
    print(f'   Using the FIRST {SUBTRACT_TIME} steps for this cutoff')

sample_size_arr = [2,4,6,8,10,12,14]

if "01" in load_path:
    num_t_steps = 62
    region_list = [[0,40],[40,61]]
    region_name = ['A-B','B-C']
    num_shape_func = 193498
elif "02" in load_path:
    num_t_steps = 78
    region_list = [[0,40],[40,77]]
    region_name = ['A-B','B-C']
    num_shape_func = 129675
elif "03" in load_path:
    num_t_steps = 71
    region_list = [[0,40],[40,70]]
    region_name = ['A-B','B-C']
    num_shape_func = 77470
elif "06" in load_path:
    num_t_steps = 213
    region_list = [[0,69],[69,148],[148,212]]
    region_name = ['A-B','B-C','C-D']
    num_shape_func = 77470
elif "11" in load_path:
    num_t_steps = 96
    region_list = [[0,60],[60,95]]
elif "16" in load_path:
    num_t_steps = 1032
    region_list = [[0,419],[419,718],[718,1030]]
    region_name = ['A-B','B-C','C-D']
elif "17" in load_path:
    num_t_steps = 1548
    # num_t_steps = 740
    # region_list = [[0,430]]
    # region_name = ['A-B']
    region_list = [[0,430],[430,739],[739,1046]]
    region_name = ['A-B','B-C','C-D']
    # region_list = [[0,430],[430,739],[739,1046],[1046,1546]]
    # region_name = ['A-B','B-C','C-D','D-E']
elif "18" in load_path:
    num_t_steps = 1043
    region_list = [[0,425],[425,735],[735,1041]]
    region_name = ['A-B','B-C','C-D']
elif "20" in load_path:
    num_t_steps = 2000
    region_list = [[0,89],[89,403],[403,1556]]
    region_name = ['A-B','B-C','C-D']
    # region_list = [[0,89],[89,403],[403,1556],[1556,1998]]
    # region_name = ['A-B','B-C','C-D','D-E']
elif "21" in load_path:
    num_t_steps = 1600
    region_list = [[0,89],[89,393],[393,1410]]
    region_name = ['A-B','B-C','C-D']
    # region_list = [[0,89],[89,393],[393,1410],[1410,1598]]
    # region_name = ['A-B','B-C','C-D','D-E']
elif "23" in load_path:
    num_t_steps = 2901
    region_list = [[0,507],[507,2500],[2500,2899]]
    region_name = ['A-B','B-C','C-D']
else:
    print('Region list not setup for simulation',flush=True)
    exit()

if LOAD_SUBSAMPLED:
    print(f'Subsampling every {sampleSize} from 0 to {num_t_steps+1}')
    load_step_arr = np.arange(0,num_t_steps+1,sampleSize)
    new_region_list = []
    for region in region_list:
        region_start_idx = np.where(load_step_arr <= region[0])[0][-1]
        if region[-1] > load_step_arr[-1]:
            region_end_idx = len(load_step_arr)-1
        else:
            region_end_idx = np.where(load_step_arr >= region[-1])[0][0]
        new_region_list.append([region_start_idx,region_end_idx])
        print(f'    Indices: {region_start_idx}, {region_end_idx} corresponding to steps {load_step_arr[region_start_idx]}, {load_step_arr[region_end_idx]}')
    region_list = new_region_list
    print(f'New region list: {region_list}')

load_suffix = ''
if LOAD_FULL == True:
    load_suffix = 'full'
elif LOAD_FULL == False:
    load_suffix = f'{num_t_steps}'

orig_suffix = load_suffix
if LOAD_COARSE:
    load_path = os.path.join(load_path,f'vsi-coarse_{coarse_mesh_idx}')
else:
    load_path = os.path.join(load_path,'vsi')
orig_load_path = load_path

if LOAD_NOISE == True:
    load_path = os.path.join(load_path,f'noise_{noise_field}_{sigma:1.0e}')
    load_path = os.path.join(load_path,f'noise-{vsiNum:02d}')
    if os.path.exists(load_path):
        print(f'Loading noisy data from VSI data set {load_path}')
    else:
        print(f'Could not find data in path {load_path}, check directory tree')
        exit()
    load_suffix += f'-noise_{noise_field}'
    # load_suffix += f'-noise_uy_{sigma:1.0e}'

if LOAD_COARSE:
    load_path = os.path.join(load_path,f'coarse_{coarse_mesh_idx}')
    if os.path.exists(load_path):
        print(f'Loading coarse data from VSI data set {load_path}')
    else:
        print(f'Could not find data in path {load_path}, check directory tree')
        exit()

if LOAD_SUBSAMPLED == True: 
    load_path = os.path.join(load_path,f'subsample_{sampleSize}')
    if os.path.exists(load_path):
        print(f'Loading subsampled data from VSI data set {load_path}')
    else:
        print(f'Could not find data in path {load_path}, check directory tree')
        exit()
    load_suffix += f'-sample_{sampleSize}'

if load_suffix == '':
    print('Issues loading with given settings, verify...')
    exit()

orig_suffix += '.npy'
load_suffix += '.npy'
print(f'Loading using the name: {load_suffix}')

save_path_alpha = os.path.join(load_path,'alpha_coef.csv')
save_path_beta = os.path.join(load_path,'beta_coef.csv')
print(f'Will save files to {save_path_alpha}')


#####################################
#            Set up log file
#####################################

logfile = os.path.join(load_path,'cvx_out.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.propagate = False
if obj_func == 1:
    fh = logging.FileHandler(logfile, mode='w')
elif obj_func == 2:
    fh = logging.FileHandler(logfile, mode='a')
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
logger.handlers = []
logger.addHandler(fh)

logger.info('Log file set up')
logger.info(f'Loading data from {load_path}')

logger.info('Loaded the bases to be used for inference:')
logger.info(f'  {num_alpha} Degradation Functions, {num_beta-1} Geometric Functions, and {num_zeta} History Functions')

logger.info('\n Using input parameters:')
logger.info(f'    Num shape func: {num_shape_func}')
logger.info(f'    Num time steps: {num_t_steps}')
logger.info(f'    Max array length: {ARRAY_CUTOFF}')
logger.info(f'    Max time steps for array length: {SUBTRACT_TIME}')

###############################################
#           First Objective Function
###############################################
if SOLVE_FIRST_OBJ:
    logger.info('\n------------------------------------------------------------')
    logger.info('            Objective Function 1')
    logger.info('------------------------------------------------------------')

    alpha_df = []
    
    A_load_path = os.path.join(load_path,f'A1_{load_suffix}')
    logger.info(f'Loading array:\n  {A_load_path}')
    A_full = np.load(A_load_path)
    logger.info(f'A array loaded, shape: {A_full.shape}')

    if LOAD_NOISE:
        print('Difference between noisy data and original:')
        A_orig_path = os.path.join(orig_load_path,f'A1_{orig_suffix}')
        A_orig = np.load(A_orig_path)
        A_diff = np.linalg.norm(A_full-A_orig,'fro')
        print(f'    A1: {A_diff}')
        del A_orig,A_diff

    logger.info('Values of ||Ax||:')
    exactFit = A_full @ exactAlpha
    logger.info(f'    Exact alphas:    {np.linalg.norm(exactFit)}')
    exactFit = A_full @ [0,1]
    logger.info(f'    Opposite alphas: {np.linalg.norm(exactFit)}')


    def solve_first_obj(A_full,sample_idxs,exactAlpha,alpha_df,dataSampling=0):
        A = A_full[sample_idxs,:]

        # Check the rank of A and dimension of the solution
        Arank = np.linalg.matrix_rank(A)
        dimComp = A.shape[1]
        if Arank != dimComp:
            print(f'Potential Error --- \n    A rank is: {Arank} compared to dimension {dimComp}')

        # Recall: Regular least squares will always recover trivial solution of 0,0, we need the constraint

        # ----------------------------------------------- SCIPY SOLVE -----------------------------------------------
        logger.info('  Solving with SciPy...') 
        def first_objective_function(xloc, Aloc):
            return np.linalg.norm(Aloc.dot(xloc))**2  # We want to minimize ||Ax||^2

        # Initial guess for x
        x0 = [0.5,0.5]

        # Constraint that sum(x) == 1 and x >= 0
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = [(0, None) for x in x0]

        # Solve the problem.
        logger.info(f'    Initial loss = {first_objective_function(x0,A)}')
        try:
            time_start = time.time()
            result = sp.minimize(first_objective_function, x0, args=(A,), method='SLSQP', bounds=bnds, constraints=cons)
            time_end = time.time()
            logger.info(f'    Final loss =   {result.fun}')
            logger.info(f'    Solved in {result.nit} iterations')
            logger.info(f'    Linear solver took {time_end-time_start:1.2e} s')
            print_xalpha(result.x)
        except:
            logger.info('    Linear solver failed, moving to next region')

        if result.status != 0:
            logger.info('  Problem solve was not optimal, ', result.message)
        else:
            temp_df = {}
            temp_df['Method'] = 'scipy'
            temp_df['Region'] = region_name[regionIdx]
            temp_df['T-Steps'] = f'{region_list[regionIdx][0]}-{region_list[regionIdx][1]}'
            if dataSampling != 0:
                temp_df['DataSampling'] = dataSampling
            for aIdx,a in enumerate(exactAlpha):
                temp_df[f'Alpha_{aIdx}'] = result.x[aIdx]
            alpha_df.append(temp_df)
            
        # ----------------------------------------------- CVXPY SOLVE -----------------------------------------------
        logger.info('  Solving with CVXPY...')
        cvx_time_steps = 20
        if A.shape[0] > cvx_time_steps*num_shape_func:
            Anew = A[0:cvx_time_steps*num_shape_func,:] # testing smaller matrices
            logger.info(f'    Reducing to {cvx_time_steps} steps for CVX solve')
        else:
            Anew = A
        from scipy.sparse import csr_matrix
        A_sparse = csr_matrix(Anew)

        # Construct the problem.
        x = cp.Variable(Anew.shape[1])

        objective = cp.Minimize(cp.sum_squares(A_sparse @ x))
        constraints = [cp.sum(x) == 1, x >= 0]
        prob = cp.Problem(objective, constraints)
        
        logger.info('    Problem set up done')

        # Solve the problem.
        try:
            time_start = time.time()
            result = prob.solve() # Alternatives: solver=cp.ECOS, OSQP, SCS
            time_end = time.time()
            logger.info(f'    Linear solver took {time_end-time_start:1.2e} s')
            print_xalpha(x.value)
        except:
            logger.info('    Linear solver failed, moving to next region')

        if prob.status != "optimal":
            logger.info('  Problem solve was not optimal, ', prob.status)
        else:
            temp_df = {}
            temp_df['Method'] = 'cvxpy'
            temp_df['Region'] = region_name[regionIdx]
            temp_df['T-Steps'] = f'{region_list[regionIdx][0]}-{region_list[regionIdx][1]}'
            if dataSampling != 0:
                temp_df['DataSampling'] = dataSampling
            for aIdx,a in enumerate(exactAlpha):
                temp_df[f'Alpha_{aIdx}'] = x.value[aIdx]
            alpha_df.append(temp_df)

        del A, Anew, A_sparse

        return alpha_df

    # Subsample each region
    regionIdx = 0
    for region in region_list:
        logger.info('\n\n------------------------------------------------------------')
        logger.info(f'Solving for coefficients in region: ({region[0]},{region[-1]})')

        if (region[-1]-region[0])*num_shape_func > ARRAY_CUTOFF:
            if USE_LAST:
                # Use last 200 steps of region
                region_start = region[-1]-SUBTRACT_TIME
                region_end = region[-1]
            else:
                # Use first 200 steps of region
                region_start = region[0]
                region_end = region[0]+SUBTRACT_TIME
            logger.info(f'  Changing region for solver to work: ({region_start},{region_end})')
        else:
            region_start = region[0]
            region_end = region[-1]

        sample_idxs = range(region_start*num_shape_func,region_end*num_shape_func)
        logger.info('\n  Attempting solve with default time step')
        # print.flush()
        alpha_df=solve_first_obj(A_full,sample_idxs,exactAlpha,alpha_df,1)

        # Subsample in time
        if SUBSAMPLE_REGIONS:
            logger.info('\n  Attempting to solve with subsampling considering original region\n')
            for subsample_size in sample_size_arr:
                # Get time steps to be sampled
                step_arr = np.arange(region[0],region[-1],subsample_size)
                num_subsampled_steps = len(step_arr)
                if num_subsampled_steps*num_shape_func > ARRAY_CUTOFF:
                    if USE_LAST:
                        # Use last 200 steps of region
                        step_arr = step_arr[num_subsampled_steps-200:num_subsampled_steps]
                    else:
                        # Use first 200 steps of region
                        step_arr = step_arr[0:200]
                    logger.info(f'    Changing subsample region: ({step_arr[0]},{step_arr[-1]})')
                if len(step_arr)*num_shape_func < ARRAY_CUTOFF:
                    logger.info(f'    Sample size {subsample_size} results in {len(step_arr)} time steps')
                    # Get indices based on step index and number of shape functions, skipping X time steps
                    first_step=True
                    for step in step_arr:
                        step_idx = np.arange(step*num_shape_func,(step*num_shape_func)+num_shape_func)
                        if first_step:
                            sample_idxs = step_idx
                            first_step = False
                        else:
                            sample_idxs = np.hstack((sample_idxs,step_idx))
                            
                    alpha_df=solve_first_obj(A_full,sample_idxs,exactAlpha,alpha_df,subsample_size)
                else:
                    logger.info(f'    Sample size {subsample_size} results in {len(step_arr)*num_shape_func} indices')
                    logger.info('    Too large for linear solve, moving to next subsample size')
                    # print.flush()

        regionIdx += 1

    del A_full # Clear up memory for second objective function
    sys.stdout.flush() # HPC Only: Force logger.info to the output file

    with open(save_path_alpha, "w", newline="") as f:
        w = csv.DictWriter(f, alpha_df[0].keys())
        w.writeheader()
        w.writerows(alpha_df)


###############################################
#           Second Objective Function
###############################################
if SOLVE_SECOND_OBJ:
    logger.info('\n------------------------------------------------------------')
    logger.info('            Objective Function 2')
    logger.info('------------------------------------------------------------')

    betazeta_df = []

    A_orig_path = os.path.join(orig_load_path,f'A2_{orig_suffix}')
    b_orig_path = os.path.join(orig_load_path,f'B_{orig_suffix}')
    A_load_path = os.path.join(load_path,f'A2_{load_suffix}')
    b_load_path = os.path.join(load_path,f'B_{load_suffix}')

    logger.info(f'Loading array:\n  {A_load_path}')
    logger.info(f'Loading array:\n  {b_load_path}')
    A_full = np.load(A_load_path)
    
    if LOAD_NOISE:
        print('Difference between noisy data and original:')
        A_orig = np.load(A_orig_path)
        A_diff = np.linalg.norm(A_full-A_orig,'fro')
        print(f'    A2: {A_diff}')
        del A_orig,A_diff

    b_full = np.load(b_load_path)
    
    if LOAD_NOISE:
        b_orig = np.load(b_orig_path)
        B_diff = np.linalg.norm(b_full-b_orig)
        print(f'    B: {B_diff}')
        del b_orig,B_diff

    logger.info(f'A and b array loaded, A shape: {A_full.shape}, b shape: {b_full.shape}')

    def solve_second_obj(A_full,b_full,sample_idxs,exactBeta,exactZeta,betazeta_df,dataSampling=0):

        A = A_full[sample_idxs,:]
        b = b_full[sample_idxs]

        # ----------------------------------------------- SCIPY SOLVE -----------------------------------------------
        logger.info('  Solving with SciPy...')
        # Function to be minimized that represents squared euclidean norm
        def second_objective_function(xloc, Aloc, bloc):
            return np.linalg.norm(Aloc.dot(xloc) - bloc)**2  # We want to minimize ||Ax-b||^2

        # Initial guess for x
        x0 = np.random.rand(A.shape[1])
        print('    Initial guess x0:', x0)

        # Bounds that x >= 0
        bnds = [(0, None) for x in x0]

        # Solve the problem.
        logger.info(f'    Initial loss = {second_objective_function(x0,A,b)}')
        try:
            time_start = time.time()
            result = sp.minimize(second_objective_function, x0, args=(A,b), method='SLSQP', bounds=bnds)
            time_end = time.time()
        except:
            logger.info('    Linear solver failed, moving to next region')
            logger.info('  Problem solve was not optimal, ', result.status)

        if result.status != 0:
                logger.info('  Problem solve was not optimal, ', result.message)
        else:
            logger.info(f'    Final loss =   {result.fun}')
            logger.info(f'    Solved in {result.nit} iterations')
            logger.info(f'    Linear solver took {time_end-time_start:1.2e} s')
            print_xbetazeta(result.x)
            
            temp_df = {}
            temp_df['Method'] = 'scipy'
            temp_df['Region'] = region_name[regionIdx]
            temp_df['T-Steps'] = f'{region_list[regionIdx][0]}-{region_list[regionIdx][1]}'
            if dataSampling != 0:
                temp_df['DataSampling'] = dataSampling
            for bIdx,b in enumerate(exactBeta):
                temp_df[f'Beta_{bIdx}'] = result.x[bIdx]
            for zIdx,z in enumerate(exactZeta):
                temp_df[f'Zeta_{zIdx}'] = result.x[num_beta+zIdx]
            betazeta_df.append(temp_df)
        # ----------------------------------------------- CVXPY SOLVE -----------------------------------------------
        A = A_full[sample_idxs,:]
        b = b_full[sample_idxs]
        
        logger.info('  Solving with CVXPY...')
        # Construct the problem.
        var_size = A.shape[1]
        x = cp.Variable(var_size)
        objective = cp.Minimize(cp.sum_squares(A @ x + b)) # Note: the + b is to avoid sign errors
        
        constraints = [x >= 0]
        prob = cp.Problem(objective, constraints)
        # prob = cp.Problem(objective)
        logger.info('    Problem set up done')

        # Solve the problem
        try:
            time_start = time.time()
            result = prob.solve()
            time_end = time.time()
        except:
            logger.info('    Linear solver (CVX) failed, moving to next region')
            logger.info('  Problem solve was not optimal, ', prob.status)
        
        if prob.status != "optimal":
            logger.info('  Problem solve was not optimal, ', prob.status)
        else:
            logger.info(f'    Linear solver took {time_end-time_start:1.2e} s')
            print_xbetazeta(x.value)

            temp_df = {}
            temp_df['Method'] = 'cvxpy'
            temp_df['Region'] = region_name[regionIdx]
            temp_df['T-Steps'] = f'{region_list[regionIdx][0]}-{region_list[regionIdx][1]}'
            if dataSampling != 0:
                temp_df['DataSampling'] = dataSampling
            for bIdx,b in enumerate(exactBeta):
                temp_df[f'Beta_{bIdx}'] = x.value[bIdx]
            for zIdx,z in enumerate(exactZeta):
                temp_df[f'Zeta_{zIdx}'] = x.value[num_beta+zIdx]
            betazeta_df.append(temp_df)
        # # ----------------------------------------------- lstsq SOLVE -----------------------------------------------
        
        del A,b

        return betazeta_df

    # Subsample each region
    regionIdx = 0
    for region in region_list:
        logger.info('\n\n------------------------------------------------------------')
        logger.info(f'Solving for coefficients in region: ({region[0]},{region[-1]})')

        if (region[-1]-region[0])*num_shape_func > ARRAY_CUTOFF:
            if USE_LAST:
                # Use last 200 steps of region
                region_start = region[-1]-SUBTRACT_TIME
                region_end = region[-1]
            else:
                # Use first 200 steps of region
                region_start = region[0]
                region_end = region[0]+SUBTRACT_TIME
            logger.info(f'  Changing region for solver to work: ({region_start},{region_end})')
        else:
            region_start = region[0]
            region_end = region[-1]

        sample_idxs = range(region_start*num_shape_func,region_end*num_shape_func)
        logger.info('\n  Attempting solve with default time step')
        
        betazeta_df=solve_second_obj(A_full,b_full,sample_idxs,exactBeta,exactZeta,betazeta_df,1)

        # Subsample in time
        if SUBSAMPLE_REGIONS:
            logger.info('\n  Attempting to solve with subsampling considering original region\n')
            for subsample_size in sample_size_arr:
                # Get time steps to be sampled
                step_arr = np.arange(region[0],region[-1],subsample_size)
                num_subsampled_steps = len(step_arr)
                if num_subsampled_steps*num_shape_func > ARRAY_CUTOFF:
                    if USE_LAST:
                        # Use last 200 steps of region
                        step_arr = step_arr[num_subsampled_steps-200:num_subsampled_steps]
                    else:
                        # Use first 200 steps of region
                        step_arr = step_arr[0:200]
                    logger.info(f'    Changing subsample region: ({step_arr[0]},{step_arr[-1]})')
                if len(step_arr)*num_shape_func < ARRAY_CUTOFF:
                    logger.info(f'    Sample size {subsample_size} results in {len(step_arr)} time steps')
                    # Get indices based on step index and number of shape functions
                    first_step=True
                    for step in step_arr:
                        step_idx = np.arange(step*num_shape_func,(step*num_shape_func)+num_shape_func)
                        if first_step:
                            sample_idxs = step_idx
                            first_step = False
                        else:
                            sample_idxs = np.hstack((sample_idxs,step_idx))

                    betazeta_df=solve_second_obj(A_full,b_full,sample_idxs,exactBeta,exactZeta,betazeta_df,subsample_size)
                else:
                    logger.info(f'  Sample size {subsample_size} results in {len(step_arr)*num_shape_func} indices')
                    logger.info('    Too large for linear solve, moving to next subsample size')
        regionIdx += 1

    del A_full, b_full # Clear up memory for second objective function


    with open(save_path_beta, "w", newline="") as f:
        w = csv.DictWriter(f, betazeta_df[0].keys())
        w.writeheader()
        w.writerows(betazeta_df)

program_end = time.time()

logger.info('\nProgram ran successfully')

total_sim_time = program_end-program_start
if total_sim_time <= 60:
    logger.info(f'\n  Main File Took {total_sim_time:1.3e} s')
elif total_sim_time > 60 and total_sim_time <= 3600:
    logger.info(f'\n  Main File Took {total_sim_time:1.3e} s = {total_sim_time/60:1.2f} min')
else:
    logger.info(f'\n  Main File Took {total_sim_time:1.3e} s = {total_sim_time/3600:1.2f} hr')
