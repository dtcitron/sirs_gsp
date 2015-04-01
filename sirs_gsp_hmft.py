import numpy as np
import scipy
import random
import cPickle as pickle
from collections import defaultdict
#import scipy.optimize as opt
import matplotlib.pyplot as plt
from sirs_gsp import *



def sirs_hmft(pk, n, r0, g = 1, rho = 1, init_i = 1, init_s = None,\
             tmax = 10, seed  = 0, debug = False):
    """
    Generate a single trajectory of a Susceptible-Infected-
    Recovered-Susceptible (SIRS) model of disease dynamics.  Exit out 
    of the simulation if the number of infecteds reaches 0 or if 
    t > tmax.
    Inputs:
        n    : Population size
        r0   : Epidemiological parameter; this is used to calculate
               the per-contact rate of transmission beta = g*r0/n
        g    : Recovery rate (gamma)
        rho  : Waning immunity rate (rho)
        init_i   : Initial number of infected nodes
        init_s   : Initial number of susceptible nodes, default None
                   If None, X = N - init_i
        tmax : Maximum total length of real time for which the 
               simulation runs
        seed : Seed for random.random
        debug: Return messages if set to True; defaults to False
    Outputs:
        t   : List of times at which observations are made
        X   : Dict of number of remaining Susceptibles, 
              indexed by times in t; {time from t:S}
        Y   : Dict of number of Infecteds, indexed by times t;
              {time from t:I}
    """
    ks = np.array(sorted(pk.keys()))
    pks = np.array([pk[k] for k in ks])
    kmean = np.dot(ks, pks)
    N = np.array(n*pks, dtype = int)
    Y = np.array(init_i*pks, dtype = int)
    if init_s != None:
        X = np.array(init_s*pks, dtype = int)
    else:
        X = N - Y
    t = [0.]
    X = {0.:X}
    Y = {0.:Y}
    random.seed(seed)
    # Check the two exit conditions, extinction and t < tmax
    while np.sum(Y[t[-1]]) and t[-1] < tmax:
        # calculate HMF
        theta = np.dot(ks, Y[t[-1]])/kmean/n
        # Calculate transition rates
        rate_SI = g*r0/kmean*ks*X[t[-1]]*theta
        rate_IR = g*Y[t[-1]]
        rate_RS = g*rho*(N - X[t[-1]] - Y[t[-1]])
        rates = np.array([rate_SI, rate_IR, rate_RS]).transpose()
        rates = np.reshape(rates, np.size(rates))
        rate_tot = np.sum(rates)
        # Calculate the time until the next event
        dt = -np.log(random.random())/rate_tot
        r = random.random()
        if debug: print rate_tot, rate_SI, rate_IR, rate_RS, r, dt, \
                        X[t[-1]], Y[t[-1]]
        t.append(t[-1]+dt)
        # Choose which event happens next
        X[t[-1]] = np.copy(X[t[-2]])
        Y[t[-1]] = np.copy(Y[t[-2]])
        index = np.digitize(np.array([r]), np.cumsum(rates)/rate_tot)[0]
        # index//3 gives the k index
        # index%3 gives the type of transition
        if index%3 == 0:
            X[t[-1]][index//3] -= 1
            Y[t[-1]][index//3] += 1
        elif index%3==1:
            Y[t[-1]][index//3] -= 1
        elif index%3==2:
            X[t[-1]][index//3] +=1
    return (t,X,Y)

def sirs_hmft_group(pk, n, r0, g = 1, rho = 1, init_i = 1, \
                    init_s = None, tmax = 10, seed  = 0, nruns = 10):
    """
    Generate a group of trajectories using the Susceptible-Infected-
    Recovered-Susceptible (SIRS) model of disease dynamics.  Each 
    trajectory is calculated in series using sirs_gsp().
    Inputs:
        pk   : Dictionary of degree distibution {k : p(k)}
        n    : Population size
        r0   : Epidemiological parameter; this is used to calculate
               the per-contact rate of transmission beta = g*r0/n
        g    : Recovery rate (gamma)
        rho  : Waning immunity rate (rho)
        init_i   : Initial number of infected nodes
        init_s   : Initial number of susceptible nodes, default None
        tmax : Maximum total length of real time for which the 
               simulation runs
        seed : Seed for random.random()
        nruns: Number of trajectories to simulate in series
    Outputs:
        t_group    : List of times of of trajectories, Array of arrays
        X_group    : List of Susceptible trajectories, stored as arrays
                        ith row is ith set of susceptibles
                        ordered by degree class
        Y_group    : List of Infected trajectories, stored as arrays
                        ith row is ith set of susceptibles
                        ordered by degree class
        To pull out the set of infected trajectories associated with the 
            3rd degree class, use Y_group[]
    """
    t_group = []
    X_group = []
    Y_group = []
    for i in range(nruns):
        (t, X, Y) = sirs_hmft(pk, n, r0, g, rho, init_i, init_s, tmax, seed)    
        t, X, Y = ex_traj(t, X, Y)
        t_group.append(t)
        X_group.append(X)
        Y_group.append(Y)
        # pick a new seed
        seed = np.random.randint(0, sys.maxint)
    return t_group, X_group, Y_group
    
def hmft_trajectory_grid(t_group, X_group, Y_group, dt = .1):
    nruns = len(t_group) # number of runs
    K = len(X_group[0]) # number of degree classes
    # Define lists of lists for t, S(t), I(t)
    t_grid = nruns*[[]]
    S_grid = []
    I_grid = []
    max_grid = 0
    for i in range(nruns):
        t_grid[i] = scipy.arange(0, t_group[i][-1] + dt, dt)
        len_grid = len(t_grid[i])
        # Map the real-time time points onto the coarse-grained time steps
        pos_grid = np.searchsorted(t_group[i],t_grid[i]) 
        S_grid.append(np.array([np.array([X_group[i][k][max(pos_grid[j]-1,0)]\
                         for j in range(len_grid)]) for k in range(K)]))
        I_grid.append(np.array([np.array([Y_group[i][k][max(pos_grid[j]-1,0)]\
                         for j in range(len_grid)]) for k in range(K)]))
    return t_grid, S_grid, I_grid
    
###----------------------------------------------------------------------------
#  Functions for phase diagram for a group of trajectories
###----------------------------------------------------------------------------
    
def hmft_sirs_diagram(pk, n, r0s, alphas, g, maxtime, dt, seed, nruns,
                      fname = 'SIRS_hmft_gsp_diagram.dat', mft = False):
    """
    Produce a group of SIRS trajectories for each given parameter
    combination using the annealed HMFT version of the SIRS Gillespie 
    dynamics. This produces a phase diagram that can be used to 
    measure the persistence of the endemic phase, statistics of 
    trajectories, and more.
    Inputs:
        pk     : Degree distribution {degree : fraction nodes with degree}
        n      : Population size
        r0s    : Array/list of R0 values in parameter space
        alphas : Array/list of alpha values in parameter space
        g      : Recovery rate (gamma)
        maxtime: Maximum total length of real time for which the 
                 simulation runs
        dt     : Time step
        seed   : Initial seed for random.random()
        nruns  : Number of trajectories to simulate in series
                 for each set of parameters
        fname  : If fname != None, write the output out to the
                 named location.  Else return the output.
        mft    : If True, determine the initial number 
    Outputs:
        data   : This is a dictionary containing the full coarse-
                 grained trajectory data for a group of trajectories
                 calculated for each parameter combination.
                 {[alpha, r0] : coarse-grained trajectory group}
                 If fname != None, write out to file at given location;
                 otherwise, return data.
    """
    data = {}
    for r0 in r0s: 
        for alpha in alphas:
            if mft:
                init_s = int(1.*n/r0)
                init_i = max(int(1.*n*alpha/(1. + alpha)*(1 - 1./r0)), n//10)
            else:
                init_s = None
                init_i = n//10
            t, x, y = sirs_hmft_group(pk, n, r0, g, alpha, \
                        init_i, init_s, maxtime, seed, nruns)
            data[alpha, r0] = hmft_trajectory_grid(t, x, y, dt)
            seed = np.random.randint(0, sys.maxint)
    if fname != None:
        f = open(fname, 'w')
        pickle.dump(data, f)
        f.close()
    else:
        return data
        
###----------------------------------------------------------------------------
        
def hmft_ensemble(t_grid, S_grid, I_grid, k = None):
    # Return the ensemble of trajectories
    # if k == None, return trajectories of total number of S and I
    # if k == an integer, return trajectories of the kth degree class S and I
    nruns = len(t_grid)
    if k == None:
        S_ens = np.array([np.sum(S_grid[i], 0) for i in range(nruns)])
        I_ens = np.array([np.sum(I_grid[i], 0) for i in range(nruns)])
    else:
        S_ens = np.array([S_grid[i][k] for i in range(nruns)])
        I_ens = np.array([I_grid[i][k] for i in range(nruns)])
    return S_ens, I_ens

def hmft_qsd(t_grid, S_grid, I_grid, k = None):
    # Return the QSD for all trajectories
    # if k == None, return QSDs of total number of S and I
    # if K == an integer, return QSDs trajectories for kth degree class
    dt = t_grid[0][1]-t_grid[0][0]
    nruns = len(t_grid)
    max_grid = np.max([len(i) for i in t_grid])
    ts = np.linspace(0, max_grid-1, max_grid)*dt;
    if k == None:
        S_qsd = np.array([[np.sum(S_grid[i], 0)[j] \
                for i in range(nruns) if j < len(S_grid[i][0])] \
                for j in range(max_grid)])
        I_qsd = np.array([[np.sum(I_grid[i], 0)[j] \
                for i in range(nruns) if j < len(I_grid[i][0])] \
                for j in range(max_grid)])
    else:
        S_qsd = np.array([[S_grid[i][k][j] \
                for i in range(nruns) if j < len(S_grid[i][0])] \
                for j in range(max_grid)])
        I_qsd = np.array([[I_grid[i][k][j] \
                for i in range(nruns) if j < len(I_grid[i][0])] \
                for j in range(max_grid)])
    return ts, S_qsd, I_qsd
    
def hmft_stats(t_grid, S_grid, I_grid, k = None):
    # Return statistics (mean and standard deviations) for QSD
    # if k == None, return stats on QSD
    # if k != None, return stats of QSD of kth degree class
    max_grid = np.max([len(i) for i in t_grid])
    t, x_qsd, y_qsd = hmft_qsd(t_grid, S_grid, I_grid, k)
    x_mean = np.array([np.mean(x_qsd[i]) for i in range(max_grid)])
    x_std = np.array([np.std(x_qsd[i]) for i in range(max_grid)])
    y_mean = np.array([np.mean(y_qsd[i]) for i in range(max_grid)])
    y_std = np.array([np.std(y_qsd[i]) for i in range(max_grid)])
    return t, x_mean, x_std, y_mean, y_std

def ex_traj(t, X, Y):
    """
    Extract all trajectories (originally output as dictionaries) and rewrite
    into a set of easily-plotted, easily accessed numpy arrays
    Inputs:
        t - list of times at which events occur
        X - dictionary {t_i: [X_1(t_i), X_2(t_i) ...]}
        Y - dictionary {t_i: [Y_1(t_i), Y_2(t_i) ...]}        
    Outputs:
        ttraj - numpy array of times at which events occur
        Xtraj - numpy array of susceptible trajectories,
                jth row is jth degree class
        Ytraj - numpy array of infected trajectories,
                jth row is jth degree class
    """
    # number of degree classes
    K = len(X[t[0]])
    ttrajs = np.array(t)
    Xtrajs = np.array([[X[i][j] for i in t] for j in range(K)])
    Ytrajs = np.array([[Y[i][j] for i in t] for j in range(K)])
    return ttrajs, Xtrajs, Ytrajs