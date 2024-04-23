import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import os
import glob
from OPE_utils import *

# Map back to the policy in discrete MDP
def convert_to_policy_table(Q):
    a_star = np.nanargmax(Q, axis=1)
    pol = np.zeros((nS, nA))
    pol[list(np.arange(nS-3)), a_star] = 1
    pol[-3:, 0] = 1
    return pol

def evaluate_helper(main_dir, data_va, pi_b_va, data_te, pi_b_te):
    ''' Iterate through all policies saved in main_dir. Calculate both validation and test performance over 100 bootstraps
    '''
    results = []
    Vs = glob.glob(os.path.join(main_dir, 'V_iter_*.npy'))
    for i in tqdm(range(len(Vs))):
        V = np.load(glob.glob(os.path.join(main_dir, 'V_iter_{}.npy'.format(i)))[0])
        Q = np.load(glob.glob(os.path.join(main_dir, 'Q_iter_{}.npy'.format(i)))[0])
        policy = pickle.load(open(os.path.join(main_dir, 'policy_iter_{}.p'.format(i)), 'rb'))
        π_e = convert_to_policy_table(Q[:k, :])

        Q = np.nan_to_num(Q[:k, :], nan = -100)
        # Iterate through the threshold for determining unimportant states
        # Identify the unimportant states
        for dQ in [-1, 0, 1, 5, 10, 15]:   
            irr_states = np.where(np.abs(Q[:, 0] - Q[:, 1]) <= dQ)[0]
            N_states = len(irr_states)
            π_e_irr = π_e.copy()

            # Calculate the bootstrapped validation performance
            π_e_irr[irr_states, :] = pi_b_va[irr_states, :]
            WIS_output_va = []
            for run in range(100):
                data_va_bootstrap = np.random.default_rng(seed=run).choice(data_va, len(data_va), replace=True)
                WIS_value_bootstrap, _, WIS_ESS_bootstrap = OPE_WIS(data_va_bootstrap, pi_b_va, π_e_irr, gamma)
                bootstrap_va = data_va_bootstrap[:, :, 3].sum() / data_va_bootstrap.shape[0]
                WIS_output_va.append((WIS_value_bootstrap, WIS_value_bootstrap-bootstrap_va, WIS_ESS_bootstrap))

            va_WIS_value = [elt[0] for elt in WIS_output_va]
            va_WIS_diff = [elt[1] for elt in WIS_output_va]  # how much the policy outperforms the behavior policy
            va_WIS_ESS = [elt[2] for elt in WIS_output_va]
            va_WIS_value_25, va_WIS_value_50, va_WIS_value_75 = np.percentile(va_WIS_value, q = [2.5, 50, 97.5])
            va_WIS_diff_25, va_WIS_diff_50, va_WIS_diff_75 = np.percentile(va_WIS_diff, q = [2.5, 50, 97.5])
            va_WIS_ESS_25, va_WIS_ESS_50, va_WIS_ESS_75 = np.percentile(va_WIS_ESS, q = [2.5, 50, 97.5])

            # Calculate the bootstrapped test performance
            π_e_irr[irr_states, :] = pi_b_te[irr_states, :]
            WIS_output_te = []
            for run in range(100):
                data_te_bootstrap = np.random.default_rng(seed=run).choice(data_te, len(data_te), replace=True)
                WIS_value_bootstrap, _, WIS_ESS_bootstrap = OPE_WIS(data_te_bootstrap, pi_b_te, π_e_irr, gamma)
                bootstrap_te = data_te_bootstrap[:, :, 3].sum() / data_te_bootstrap.shape[0]
                WIS_output_te.append((WIS_value_bootstrap, WIS_value_bootstrap-bootstrap_te, WIS_ESS_bootstrap))

            te_WIS_value = [elt[0] for elt in WIS_output_te]
            te_WIS_diff = [elt[1] for elt in WIS_output_te]
            te_WIS_ESS = [elt[2] for elt in WIS_output_te]
            te_WIS_value_25, te_WIS_value_50, te_WIS_value_75 = np.percentile(te_WIS_value, q = [2.5, 50, 97.5])
            te_WIS_diff_25, te_WIS_diff_50, te_WIS_diff_75 = np.percentile(te_WIS_diff, q = [2.5, 50, 97.5])
            te_WIS_ESS_25, te_WIS_ESS_50, te_WIS_ESS_75 = np.percentile(te_WIS_ESS, q = [2.5, 50, 97.5])

            # Save results
            results.append([i, dQ, N_states, va_WIS_value_25, va_WIS_value_50, va_WIS_value_75, va_WIS_diff_25, va_WIS_diff_50, va_WIS_diff_75, va_WIS_ESS_25, va_WIS_ESS_50, va_WIS_ESS_75,
                                             te_WIS_value_25, te_WIS_value_50, te_WIS_value_75, te_WIS_diff_25, te_WIS_diff_50, te_WIS_diff_75, te_WIS_ESS_25, te_WIS_ESS_50, te_WIS_ESS_75])

    df = pd.DataFrame(results, columns = ['iter', 'dQ', 'N_irr_states', 'val_value_25', 'val_value_50', 'val_value_75', 'val_diff_25', 'val_diff_50', 'val_diff_75', 'val_ESS_25', 'val_ESS_50', 'val_ESS_75', 
                                          'test_value_25', 'test_value_50', 'test_value_75', 'test_diff_25', 'test_diff_50', 'test_diff_75', 'test_ESS_25', 'test_ESS_50', 'test_ESS_75'])
    df.to_csv(os.path.join(main_dir, '_results_irr.csv'), index = False)
    
def evaluate_policy_for_each_setting(df_va, df_te, main_dir, k = 100):
    '''
    Wrapper function for evaluating the policies on the validation and test set. 
        df_va  : validation trajectory data in the format of `ens.csv`
        df_te  : test trajectory data in the format of `ens.csv`
        main_dir: directory where all policies are saved
        k  : number of discrete states
    '''
    df_va = df_va[['hosp_id', 'window_id', 'state', 'action', 'reward', 'next_state']]
    df_va.columns = ['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']
    df_va = df_va[~((df_va['Action'] == 1) & (df_va['NextState'] >= k))].copy()
    df_va.loc[(df_va['NextState'] >= k), 'Action'] = -1
    pi_b_va = compute_behavior_policy(df_va)
    data_va = format_data_tensor(df_va)
    
    df_te = df_te[['hosp_id', 'window_id', 'state', 'action', 'reward', 'next_state']]
    df_te.columns = ['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']
    df_te = df_te[~((df_te['Action'] == 1) & (df_te['NextState'] >= k))].copy()
    df_te.loc[(df_te['NextState'] >= k), 'Action'] = -1
    pi_b_te = compute_behavior_policy(df_te)
    data_te = format_data_tensor(df_te)

    evaluate_helper(main_dir, data_va, pi_b_va, data_te, pi_b_te)
    
def main():
    ''' Dummy example for how these functions will be utilized in practice '''
    evaluate_policy_for_each_setting(df_va, df_te, main_dir, k)
