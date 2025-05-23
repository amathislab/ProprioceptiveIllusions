import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv
import opensim as osim
from directory_paths import SAVE_DIR, PARENT_DIR

with h5py.File(os.path.join(SAVE_DIR, "data/cleaned_smooth/flag_pcr_train.hdf5"), 'r') as f:
    muscle_lengths = f['muscle_lengths'][()]
    muscle_velocities = f['muscle_velocities'][()]
    muscle_accelerations = f['muscle_accelerations'][()]

MUSCLE_SCALE = 1000

MUSCLE_NAMES_FOR_ELBOW = ['CORB', 'DELT1', 'DELT2', 'DELT3', 'INFSP', 'LAT1', 'LAT2', 'LAT3', 'PECM1', 
                          'PECM2', 'PECM3', 'SUBSC', 'SUPSP', 'TMAJ', 'TMIN', 'ANC', 'BIClong', 
                          'BICshort', 'BRA', 'BRD', 'ECRL', 'PT', 'TRIlat', 'TRIlong', 'TRImed']


PATH_TO_OSIM = os.path.join(SAVE_DIR, "data/raw/MOBL_ARMS_41_seb_writing_pos.osim")
# ------------------------------------------------------------------------------------------------------

def get_optimal_fiber_length():
    '''
    Returns a numpy array of shape (25,) containing the 
    optimal fiber lengths of the muscles of interest
    '''

    # initialize the arm
    model = osim.Model(PATH_TO_OSIM)
    init_state = model.initSystem()
    model.equilibrateMuscles(init_state)

    # get the muscles
    muscle_set = model.getMuscles()

    # to save the lengths
    optimal_length = np.zeros(25)

    # saves the muscle's optimal fiber lengths
    for muscle in muscle_set:
        if muscle.getName() in MUSCLE_NAMES_FOR_ELBOW:
            optimal_length[MUSCLE_NAMES_FOR_ELBOW.index(muscle.getName())] = (muscle.get_optimal_fiber_length())*MUSCLE_SCALE

    return optimal_length

optimal_length = get_optimal_fiber_length()

normalized_muscle_lengths = np.zeros_like(muscle_lengths)
normalized_muscle_velocities = np.zeros_like(muscle_velocities)
normalized_muscle_accelerations = np.zeros_like(muscle_accelerations)

for i in range(25):
    normalized_muscle_lengths[:, i, :] = muscle_lengths[:, i, :] / optimal_length[i] - 1
    normalized_muscle_velocities[:, i, :] = muscle_velocities[:, i, :] / optimal_length[i]
    normalized_muscle_accelerations[:, i, :] = muscle_accelerations[:, i, :] / optimal_length[i]

def spindle_transfer_function(length, velocity, acceleration, k_l, k_v, e_v, k_a, k_c):
    firing_rate = k_l * length + k_v * np.sign(velocity) * np.abs(velocity) ** e_v + k_a * acceleration + k_c
    firing_rate = np.clip(firing_rate, 0, None)
    return firing_rate

def objective(params, length_data, velocity_data, acceleration_data, lambda_reg, target_zero, target_fmax):
    k_l, k_v, e_v, k_a, k_c = params
    rates = spindle_transfer_function(length_data, velocity_data, acceleration_data, k_l, k_v, e_v, k_a, k_c)
    Fmax = np.max(rates)
    
    # Calculate fraction of zeros to penalize sparse firing
    frac_zero = np.mean(rates == 0)
    
    # Primary objective: get max firing ~ 80
    # Secondary objective: minimize fraction of zeros
    cost = (Fmax - target_fmax)**2 + lambda_reg * np.abs(frac_zero - target_zero) ** 2
    return cost

from scipy.optimize import differential_evolution

spindle_i_a_range = (50, 180)
spindle_ii_range = (20, 50)
constant_e_v = 1

lambda_reg = 1000

parameters_per_muscle = {}

for m in range(25):

    lengths = normalized_muscle_lengths[:2000, m, :]
    velocities = normalized_muscle_velocities[:2000, m, :]
    accelerations = normalized_muscle_accelerations[:2000, m, :]

    k_l_list = []
    k_v_list = []
    e_v_list = []
    k_a_list = []
    k_c_list = []
    frac_zero_list = []
    max_rate_list = []

    for i in range(30):
        print(f'Optimizing muscle {m}, iteration {i}')

        target_zero = np.random.uniform(0.00, 0.3)
        target_fmax = np.random.uniform(spindle_i_a_range[0], spindle_i_a_range[1])

        bounds = [
            (0, target_fmax),   # k_l range
            (0, target_fmax * 0.5),   # k_v range
            (constant_e_v, constant_e_v),  # e_v range (must be positive)
            (0, target_fmax * 0.1),  # k_a range
            (0, target_fmax * 0.1)  # k_c range
        ]
        
        result = differential_evolution(
            objective, 
            bounds,
            args=(lengths, velocities, accelerations, lambda_reg, target_zero, target_fmax),
            seed=i,
            maxiter=20,
            tol=1,
        )

        k_l, k_v, e_v, k_a, k_c = result.x
        rates = spindle_transfer_function(lengths, velocities, accelerations, k_l, k_v, e_v, k_a, k_c)
        max_rate = np.max(rates)
        frac_zero = np.mean(rates == 0)

        print(f'Optimal parameters: {result.x}')
        print(f'Max firing rate: {max_rate} (target: {target_fmax})')
        print(f'Fraction of zero firing rates: {frac_zero} (target: {target_zero})')

        k_l_list.append(k_l)
        k_v_list.append(k_v)
        e_v_list.append(e_v)
        k_a_list.append(k_a)
        k_c_list.append(k_c)
        max_rate_list.append(max_rate)
        frac_zero_list.append(frac_zero)


    parameters_per_muscle[m] = {
        'k_l': k_l_list,
        'k_v': k_v_list,
        'e_v': e_v_list,
        'k_a': k_a_list,
        'k_c': k_c_list,
        'max_rate': max_rate_list,
        'frac_zero': frac_zero_list
    }

    print(f'Finished muscle {m}')


with open(os.path.join(SAVE_DIR, "coefficients_i_a.csv"), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Muscle', 'k_l', 'k_v', 'e_v', 'k_a', 'k_c', 'max_rate', 'frac_zero'])
    for m in range(25):
        writer.writerow([m, 
                         parameters_per_muscle[m]['k_l'], 
                         parameters_per_muscle[m]['k_v'], 
                         parameters_per_muscle[m]['e_v'], 
                         parameters_per_muscle[m]['k_a'], 
                         parameters_per_muscle[m]['k_c'],
                         parameters_per_muscle[m]['max_rate'],
                         parameters_per_muscle[m]['frac_zero']])
        
for m in range(25):

    lengths = normalized_muscle_lengths[:2000, m, :]
    velocities = normalized_muscle_velocities[:2000, m, :]
    accelerations = normalized_muscle_accelerations[:2000, m, :]

    k_l_list = []
    k_v_list = []
    e_v_list = []
    k_a_list = []
    k_c_list = []
    frac_zero_list = []
    max_rate_list = []

    for i in range(30):
        print(f'Optimizing muscle {m}, iteration {i}')

        target_zero = np.random.uniform(0.00, 0.3)
        target_fmax = np.random.uniform(spindle_ii_range[0], spindle_ii_range[1])

        bounds = [
            (0, target_fmax),   # k_l range
            (0, target_fmax * 0.5),   # k_v range
            (constant_e_v, constant_e_v),  # e_v range (must be positive)
            (0, target_fmax * 0.0),  # k_a range
            (0, target_fmax * 0.1)  # k_c range
        ]
        
        result = differential_evolution(
            objective, 
            bounds,
            args=(lengths, velocities, accelerations, lambda_reg, target_zero, target_fmax),
            seed=i,
            maxiter=20,
            tol=1,
        )

        k_l, k_v, e_v, k_a, k_c = result.x
        rates = spindle_transfer_function(lengths, velocities, accelerations, k_l, k_v, e_v, k_a, k_c)
        max_rate = np.max(rates)
        frac_zero = np.mean(rates == 0)

        print(f'Optimal parameters: {result.x}')
        print(f'Max firing rate: {max_rate} (target: {target_fmax})')
        print(f'Fraction of zero firing rates: {frac_zero} (target: {target_zero})')

        k_l_list.append(k_l)
        k_v_list.append(k_v)
        e_v_list.append(e_v)
        k_a_list.append(k_a)
        k_c_list.append(k_c)
        max_rate_list.append(max_rate)
        frac_zero_list.append(frac_zero)


    parameters_per_muscle[m] = {
        'k_l': k_l_list,
        'k_v': k_v_list,
        'e_v': e_v_list,
        'k_a': k_a_list,
        'k_c': k_c_list,
        'max_rate': max_rate_list,
        'frac_zero': frac_zero_list
    }

    print(f'Finished muscle {m}')

with open(os.path.join(SAVE_DIR, "coefficients_ii.csv"), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Muscle', 'k_l', 'k_v', 'e_v', 'k_a', 'k_c', 'max_rate', 'frac_zero'])
    for m in range(25):
        writer.writerow([m, 
                         parameters_per_muscle[m]['k_l'], 
                         parameters_per_muscle[m]['k_v'], 
                         parameters_per_muscle[m]['e_v'], 
                         parameters_per_muscle[m]['k_a'], 
                         parameters_per_muscle[m]['k_c'],
                         parameters_per_muscle[m]['max_rate'],
                         parameters_per_muscle[m]['frac_zero']])