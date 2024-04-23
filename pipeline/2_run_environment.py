import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import os
import glob
import random
from collections import defaultdict

###################################################################################
#######################         Helper Functions        ###########################
###################################################################################

def get_taken_actions(state, transition_counts):
    a_ns = transition_counts[state,:,:]  # all the times that state was visited
    a_ns = a_ns.sum(axis=1) # how many times each action was taken
    action_mask = a_ns > 0
    return np.where(action_mask==True)[0] # Return indices of actions that were taken at least once

# converts trajectories from pandas df to numpy array
def transform_array(data):
    """
    Convert trajectories from pandas df to numpy array. End of episodes are padded to have the same length.
    Padding after end of episode is (s,a,r,s') = (-1,-1,0,-1)
    Returns:
      - trajectories: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
    """
    max_length = 0
    for i, g in data.groupby('hosp_id'):
        if max_length < len(g['state']):
            max_length = len(g['state'])
    
    trajectories=[]
    for i, g in tqdm(data.groupby('hosp_id')):
        s_list = np.array(g['state'])
        a_list = np.array(g['action'])
        r_list = np.array(g['reward'])
        sx_list = np.array(g['next_state'])
        
        pad_length = max_length - len(s_list)
        s_list = np.pad(s_list, (0,pad_length), mode='constant', constant_values=(0,-1))
        a_list = np.pad(a_list, (0,pad_length), mode='constant', constant_values=(0,-1))
        r_list = np.pad(r_list, (0,pad_length), mode='constant', constant_values=(0,0))
        sx_list = np.pad(sx_list, (0,pad_length), mode='constant', constant_values=(0,-1))
        
        t_list = np.arange(len(s_list))
        
        trajectory = np.stack([t_list,s_list,a_list,r_list,sx_list], axis=0)
        trajectory = trajectory.T
        trajectories.append(trajectory)
    
    return np.stack(trajectories, axis=0)

class Transitions:
    """Class for estimated trainsitions from data"""
    
    def __init__(self, nS=103, nA=2):
        """
        Initialize transition counts and probabilities to be of the right shape
        """
        self.nS = nS
        self.trans_cts = np.zeros((nS, nA, nS))  # (s, a, s')
        self.trans_prb = np.zeros((nS, nA, nS))  # (s, a, s')
        
    def fit(self, trajectories):
        """
        Estimate transition probabilites and counts from trajectories
        Inputs:
          - trajectories: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        """
        N, T, _ = trajectories.shape
        # count transition occurences in trajectories
        for i in range(N):
            for t in range(T):
                _, s, a, r, s_ = trajectories[i, t]
                self.trans_cts[s, a, s_] += 1
                # if next state is a terminal state, add transition from terminal to absorbing state, and from absorbing state to absorbing state
                if s_ == self.nS-2 or s_ == self.nS-3:  
                    self.trans_cts[s_, a, self.nS-1] += 1
                    self.trans_cts[self.nS-1, a, self.nS-1] += 1
                if s == -1: # reached end of trajectory
                    break
        # MLE for probabilities
        self.trans_prb = self.trans_cts / np.sum(self.trans_cts, axis=2)[:,:,np.newaxis]
        self.trans_prb[np.isnan(self.trans_prb)] = 0.0
        
class EnsTransitions:
    """Random Forest of transitions estimated from data."""
    
    def __init__(self, trajectories, threshold, ensemble_size=100, nS=103, nA=2, dist='tvd'):
        self.ensemble = []
        self.threshold = threshold
        self.ensemble_size = ensemble_size
        self.nS = nS
        self.nA = nA
        self.dist = dist
        for i in tqdm(range(ensemble_size)):
            traj_bs = self._bootstrap(trajectories)
            trans = Transitions(nS, nA)
            trans.fit(traj_bs)
            self.ensemble.append(trans)
        print("Computing unknown state/actions...")
        self.unknown_mask = self._compute_unknown_sa()
        print("Done!")
    
    def _bootstrap(self, data):
        N, T, _ = data.shape
        idx = np.random.choice(N, size=(N), replace=True)
        return data[idx]
    
    def _compute_unknown_sa(self):
        """
        Return a boolean mask of shape (nS, nA) corresponding to unknown state-action pairs.
        Determines unknown mask through total variation distance.
        """
        # 1 indicates unknown, 0 indicates known
        unknown_mask = np.zeros((self.nS, self.nA), dtype=np.uint8)
        if self.dist == 'tvd':
            # Compute distance based upon total variation distance.
            for s in tqdm(range(self.nS)):
                for a in range(self.nA):
                    max_diff = 0.0
                    for i in range(self.ensemble_size):
                        for j in range(i, self.ensemble_size):
                            trans_A = self.ensemble[i]
                            trans_B = self.ensemble[j]
                            diff = np.abs(trans_A.trans_prb[s,a,:] - trans_B.trans_prb[s,a,:])
                            if np.max(diff) > max_diff:
                                max_diff = np.max(diff)
                    if max_diff == 0.0: # If there is no difference, it likely means that the [s, a] was never visited. The only exception is when S are the terminal states. We deal with this below.
                        unknown_mask[s,a] = 1
                    
                    if max_diff > self.threshold:
                        unknown_mask[s,a] = 1
            return unknown_mask
        else:
            raise NotImplementedError
    
    def get_unknowns(self):
        return self.unknown_mask

###################################################################################
#######################          Main Functions         ###########################
###################################################################################

def create_transitions(traj, save_dir, k = 100):
    ''' 
    Create the transition matrix and reward function from the trajectories
        traj: trajectory data with state, action, reward, and next state. Look at `data/env.csv' for example.
        k   : number of discete states
    '''
    S, A = k+3, 2

    # iterate over each hosp_id, window_id and create transition matrix
    transition_cts = np.zeros((S, A, S)).astype(int) # shape=(state, action, next_state)
    for index, row in traj.iterrows():
        s = row['state']
        a = row['action']
        s_ = row['next_state']

        transition_cts[s, a, s_] += 1
        # if next state is a terminal state, add transition from terminal to absorbing state, and from absorbing state to absorbing state
        if s_ == S-2 or s_ == S-3:  
            transition_cts[s_, a, S-1] += 1
            transition_cts[S-1, a, S-1] += 1
            
    transition_cts = transition_cts.astype(int)
    # normalize to probabilities w.r.t number of times we entered the state-action pair
    transition_prob = transition_cts / np.sum(transition_cts, axis=2)[:,:,np.newaxis]
    transition_prob[np.isnan(transition_prob)] = 0.0
    
    # rewards matrix
    rewards = np.zeros(S)
    rewards[S-3] = 100.0   # Final outcome = lived
    rewards[S-2] = -100.0  # Final outcome = death
    rewards[S-1] = 0   # absorbing state
    
    # Save transition counts, probs, and rewards matrices
    if not os.path.exists(save_dir + 'VI'):
        os.makedirs(save_dir + 'VI')
        
    np.savez(save_dir + 'VI/transitions.npz', counts=transition_cts, probs=transition_prob)
    np.savez(save_dir + 'VI/rewards.npz', rewards=rewards)

def create_dictionary(save_dir, k = 100):
    '''
    Convert the transition data into a dictionary. 
    Use the transitions.npz generated from `create_transitions()`.
    '''
    S, A = k+3, 
    with np.load(save_dir + 'VI/transitions.npz') as f:
        trans_ct = f['counts']
        trans_prob = f['probs']
        
    mca = np.argmax(trans_ct.sum(axis=(0, 2))) # most common action
    # transition distribution is a dictionary of { (s,a): [ (prob, s', r, done) ] }
    P = defaultdict(lambda: defaultdict(list))
    for s in tqdm(range(trans_prob.shape[0])):
        taken_actions = get_taken_actions(s, trans_ct)
        # no valid actions
        if taken_actions.size == 0:
            print("No valid actions for state {}".format(s))
        for a in taken_actions:
            for s_ in range(trans_prob.shape[2]):
                prob = trans_prob[s, a, s_]
                if prob != 0:
                    # terminal rewards {-100, +100}, all intermediate rewards 0
                    # done is only true if state is absorbing state
                    P[s][a].append((prob, s_, int(s_ in [S-3, S-2]) * (100 if int(s_ == S-3) else -100), (s_ in [S-1]))) 
                    
    with open(save_dir + 'VI/MDP_P.pkl', 'wb') as f:
        pickle.dump(dict(P), f)

def create_uncertainty_transitions(traj, threshold_list, save_dir, k = 100):
    '''
    Create transition matrix and reward function for trajectories when using pessmistic MDP.
        traj: trajectory data with state, action, reward, and next state. Look at `data/env.csv' for example.
        threshold_list : thresholds used to determine uncertainty.
        k   : number of discete states
    '''
    S, A = k+3, 2
    traj = transform_array(traj)
    
    if not os.path.exists(save_dir + 'pMDP'):
        os.makedirs(save_dir + 'pMDP')
            
    for threshold in threshold_list:
        ens_transitions = EnsTransitions(traj, threshold, ensemble_size=100, nS=S, nA=A)
        mask = ens_transitions.get_unknowns()
        print(threshold, np.sum(mask))
        
        # pessimistic transition function
        transitions = Transitions(nS=S, nA=A)
        transitions.fit(traj)
        trans_prb = transitions.trans_prb
        trans_cts = transitions.trans_cts

        pess_prb = np.zeros((S, A, S))  # pessimistic transition probs 
        pess_cts = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                if mask[s,a] == 1 and s < S-3: #
                    pess_prb[s, a, S-2] = 1  # 100% chance transition to 'death' state
                    pess_cts[s, a, S-2] = np.sum(trans_cts[s, a])
                else:
                    pess_prb[s, a] = trans_prb[s, a]  # same as original transitions
                    pess_cts[s, a] = trans_cts[s, a]
                    
        np.savez(save_dir + 'pMDP/transitions_threshold_{}.npz'.format(threshold), counts=pess_cts, probs=pess_prb)
        
    # pessimistic reward function
    pess_rewards = np.zeros(S)
    pess_rewards[S-3] = 100   # survive
    pess_rewards[S-2] = -100  # death or unknown state penalty
    pess_rewards[S-1] = 0     # absorbing states
    np.savez(save_dir + 'pMDP/rewards.npz', rewards=pess_rewards)

def create_pMDP_dictionary(threshold_list, save_dir, k = 100):
    '''
    Convert the pMDP transition data into a dictionary. 
    Use the transitions_threshold_{}.npz generated from `create_uncertainty_transitions()`.
    '''
    S, A = k+3, 2
    for threshold in threshold_list:
        with np.load(save_dir + 'pMDP/transitions_threshold_{}.npz'.format(threshold)) as f:
            trans_ct = f['counts']
            trans_prob = f['probs']
            
        mca = np.argmax(trans_ct.sum(axis=(0, 2))) # most common action
        # transition distribution is a dictionary of { (s,a): [ (prob, s', r, done) ] }
        P = defaultdict(lambda: defaultdict(list))
        for s in tqdm(range(trans_prob.shape[0])):
            taken_actions = get_taken_actions(s, trans_ct)
            # no valid actions
            if taken_actions.size == 0:
                print("No valid actions for state {}".format(s))
            for a in taken_actions:
                for s_ in range(S):
                    prob = trans_prob[s, a, s_]
                    if prob != 0:
                        P[s][a].append((prob, s_, int(s_ in [S-3, S-2]) * (100 if int(s_ == S-3) else -100), (s_ in [S-1]))) 
        
        with open(save_dir + 'pMDP/pMDP_P_threshold_{}.pkl'.format(threshold), 'wb') as f:
            pickle.dump(dict(P), f)
            
def main():
    ''' Dummy example for how these functions will be utilized in practice '''           
    threshold_list = [0.1, 0.15, 0.2, 0.3]
    create_transitions()
    create_dictionary()
    create_uncertainty_transitions()
    create_pMDP_dictionary()
