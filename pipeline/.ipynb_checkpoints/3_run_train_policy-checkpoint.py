import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import os
import glob

########################################################################
###################    Generate Q action masks      ####################
########################################################################
def generate_Q_masks(save_dir, BCQ = True, pMDP = False, k = 100):
    ''' Generate the Q action masks based on the policy constraints '''
    
    if BCQ and not pMDP:  ## VI_BCQ
        with np.load(save_dir + 'transitions.npz') as f:
            trans_ct = f['counts']

        for tau in [0.0, 0.1, 0.2, 0.3, 0.4]:
            policy_counts = trans_ct.sum(axis=2)
            max_counts = np.max(policy_counts, axis=1)[:, None]  # number of times the action that was taken most frequently was taken
            Q_mask_relative = (policy_counts > (max_counts * tau))
            Q_mask_absolute = (policy_counts > 5)
            Q_mask = np.logical_and(Q_mask_relative, Q_mask_absolute)

            # For states without any available actions, allow the most frequent action
            for s in range(policy_counts.shape[0]):
                if Q_mask[s].sum() == 0:
                    Q_mask[s, policy_counts[s].argmax()] = True
                    
            np.save(save_dir + 'action_mask_tau_{}.npy'.format(tau), Q_mask)
    
    if pMDP and not BCQ: ## pMDP
        for threshold in [0.1, 0.15, 0.2, 0.3]:
            with np.load(save_dir + 'transitions_threshold_{}.npz'.format(threshold)) as f:
                trans_ct = f['counts']

            policy_counts = trans_ct.sum(axis=2)
            Q_mask = (policy_counts > 5)

            # For states without any available actions, allow the most frequent action
            for s in range(policy_counts.shape[0]):
                if Q_mask[s].sum() == 0:
                    Q_mask[s, policy_counts[s].argmax()] = True
            # check every state has at least one action, |A(s)| >= 1
            assert (Q_mask.sum(axis=1) > 0).all()
        
            np.save(save_dir + 'action_mask_threshold_{}.npy'.format(threshold), Q_mask)
            
    if pMDP and BCQ: ## pMDP_BCQ
        threshold = 0.1
        with np.load(save_dir + 'transitions_threshold_{}.npz'.format(threshold)) as f:
            trans_ct = f['counts']
            trans_prob = f['probs']

        for tau in [0.1, 0.2, 0.3, 0.4]:
            policy_counts = trans_ct.sum(axis = 2)
            max_counts = np.max(policy_counts, axis=1)[:, None]  # number of times the action that was taken most frequently was taken
            Q_mask_relative = (policy_counts > (max_counts * tau))
            Q_mask_absolute = (policy_counts > 5)
            Q_mask = np.logical_and(Q_mask_relative, Q_mask_absolute)

            # For states without any available actions, allow the most frequent action
            for s in range(policy_counts.shape[0]):
                if Q_mask[s].sum() == 0:
                    Q_mask[s, policy_counts[s].argmax()] = True

            np.save(save_dir + 'action_mask_threshold_{}_BCQ_tau_{}.npy'.format(threshold, tau), Q_mask)
    
########################################################################
###################          Train Policies         ####################
########################################################################
def VI_w_intermediate_save(main_dir, P, Q_mask):
    
    nS, nA = Q_mask.shape
    gamma = 0.99
    theta = 1e-10

    # Value iteration
    V = np.zeros(nS)
    for i in tqdm(itertools.count()):
        delta = 0.0
        for s in range(nS):
            old_v = V[s]
            # V[s] = max {a} sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            Q_s = np.zeros(nA)
            for a in P[s]:
                Q_s[a] = sum(p * (r + (0 if done else gamma * V[s_])) for p, s_, r, done in P[s][a])
            Q_s[~Q_mask[s]] = np.nan
            new_v = np.nanmax(Q_s)
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))

            np.save(os.path.join(main_dir, 'V_iter_{}.npy'.format(i)), V)
            
        if delta < theta:
            break

def save_policy(main_dir, P, Q_mask):

    nS, nA = Q_mask.shape
    gamma = 0.99
    
    Vs = glob.glob(os.path.join(main_dir, 'V_iter_*.npy'))
    for i in range(len(Vs)):
        V_file = glob.glob(os.path.join(main_dir, 'V_iter_{}.npy'.format(i)))[0]
        V = np.load(V_file)
        
        policy = np.zeros(nS, dtype=np.int)
        Q = np.zeros((nS, nA))
        for s in tqdm(range(nS)):
            for a in P[s]:
                Q[s,a] = sum(p * (r + (0 if done else gamma * V[s_])) for p, s_, r, done in P[s][a])
            Q[s,~Q_mask[s]] = np.nan
            best_action = np.nanargmax(Q[s])
            if best_action is None:
                policy[s] = 0
            else:
                policy[s] = best_action

        np.save(os.path.join(main_dir, 'Q_iter_{}.npy'.format(i)), Q)
        with open(os.path.join(main_dir, 'policy_iter_{}.p'.format(i)), 'wb') as f:
            pickle.dump(policy, f)
            
def train_policy(main_dir, save_dir, k = 100):

    P = pickle.load(open(main_dir + 'MDP_P.pkl', "rb"))
    Q_mask = np.load(main_dir + 'action_mask_tau_{}.npy'.format(tau))
    VI_w_intermediate_save(save_dir, P, Q_mask)
    save_policy(save_dir, P, Q_mask)

#########################################################################################    
def main():
    ''' Dummy example for how these functions will be utilized in practice '''
    
    generate_Q_masks(save_dir, BCQ, pMDP, k)
    train_policy(main_dir, save_dir, BCQ, pMDP, k)

