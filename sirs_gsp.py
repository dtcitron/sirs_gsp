#
#  sirs_gsp.py
#  
#
#  Created by Daniel Citron on 5/20/14.
#  Most recent update: 11/26/14 (for PEP compliance)
#   
#  sirs_gsp = SIRS simulations with Gillespie dynamics
#  
#  This module allows us to implement continuous-time stochastic 
#  simulations of SIR-type disease dynamics.  We allow for SI, SIS,
#  SIR, and SIRS models in fully mixed systems.  The simulations use
#  Gillespie's exact algorithm.
#
#  Thanks to Sarabjeet Singh, who provided the inspiration for this
#  code with Gillespie_messy.py
#
#  For large networks (>500 nodes), we use Scipy's sparse matrix
#  methods to perform the simulations.  This has been found to
#  greatly improve the speed of the simulations.  This works best
#  for graphs with a low density of edges.
#
#  Updates for later:
#  1. I have also attempted to extend the Gillespie algorithm for SIR-type
#     simulations on complex networks (given a connectivity graph), but
#     this is not yet optimized.
#  2. Is there a simple way to use map() to calculate groups of 
#     trajectories in parallel, rather than in series?

import numpy as np
import scipy
import random
import cPickle as pickle
from collections import defaultdict
import scipy.optimize as opt

# First calculate a group of trajectories:
#
# N = 500
# ii = .9*N
# ttraj, Xtraj, Ytraj = SIRS_group(N, R0 = 2, g = 1, rho = .5, ii = ii, tmax = 200, seed = 2, N_runs=10)
#
# Now map the trajectories onto a grid, coarse-grained in time:
#
# t_grid, S_grid, I_grid = GSP_grid(N, ttraj, Xtraj, Ytraj, dt = .1)
#
# Now we can extract statistics from the trajectories:
#
# t, Sm, Im, Sstd, Istd = avg_profile(t_grid, S_grid, I_grid)
#

###----------------------------------------------------------------------------
#  Functions for calculating single trajectories
###----------------------------------------------------------------------------

def si_gsp(n, r0, ii = 1, tmax = 10, seed  = 0):
    """
    Generate a single trajectory of a Susceptible-Infected (SI)
    model of disease dynamics.  Exit out of the simulation if the 
    number of infecteds reaches 0 or if t > tmax.
    Inputs:
        n   : Population size
        r0  : Epidemiological parameter; this is used to calculate
              the per-contact rate of transmission beta = g*r0/n,
              where we pick g=1 for SI
        ii  : Initial number of infected nodes
        tmax: Maximum total length of real time for which the 
              simulation runs
        seed: Seed for random.random
    Outputs:
        t   : List of times at which observations are made
        X   : Dict of number of remaining susceptibles, 
              indexed by times in t; {time from t:S}
    """
    t = [0.]
    X = {0.:float(n-ii)}
    Y = {0.:float(ii)}
    random.seed(seed)
    # Check the two exit conditions, extinction and t < tmax
    while X[t[-1]] and t[-1] < tmax:
        # Calculate transition rates
        rate_SI = r0*X[t[-1]]*Y[t[-1]]/n
        rate_tot = rate_SI
        # Calculate the time until the next event
        dt = -np.log(random.random())/rate_tot 
        t.append(t[-1]+dt)
        # Choose which event happens next
        if (random.random() <= rate_SI/rate_tot):
            X[t[-1]] = X[t[-2]]-1
            Y[t[-1]] = Y[t[-2]]+1
        else:
            X[t[-1]] = X[t[-2]]
            Y[t[-1]] = Y[t[-2]]     
    return (t,X,Y)

def sis_gsp(n, r0, g = 1, ii = 1, tmax = 10, seed  = 0):
    """
    Generate a single trajectory of a Susceptible-Infected-Susceptible
    (SIS) model of disease dynamics.  Exit out of the simulation if 
    the number of infecteds reaches 0 or if t > tmax.
    Inputs:
        n   : Population size
        r0  : Epidemiological parameter; this is used to calculate
              the per-contact rate of transmission beta = g*r0/n
        g   : Recovery rate (gamma)
        ii  : Initial number of infected nodes
        tmax: Maximum total length of real time for which the 
              simulation runs
        seed: Seed for random.random
    Outputs:
        t   : List of times at which observations are made
        X   : Dict of number of remaining Susceptibles, 
              indexed by times in t; {time from t:S}
        Y   : Dict of number of Infecteds, indexed by times t;
              {time from t:I}
    """
    t = [0.]
    X = {0.:float(n-ii)}
    Y = {0.:float(ii)}
    random.seed(seed)
    # Check the two exit conditions, extinction and t < tmax
    while Y[t[-1]] and t[-1] < tmax:
        # Calculate transition rates
        rate_SI = g*r0*X[t[-1]]*Y[t[-1]]/n
        rate_IR = g*Y[t[-1]]
        rate_tot = rate_SI + rate_IR
        # Calculate the time until the next event
        dt = -np.log(random.random())/rate_tot 
        t.append(t[-1]+dt)
        # Choose which event happens next
        if (random.random() <= rate_SI/rate_tot): # calculate 
            X[t[-1]] = X[t[-2]]-1
            Y[t[-1]] = Y[t[-2]]+1
        else:
            X[t[-1]] = X[t[-2]]+1
            Y[t[-1]] = Y[t[-2]]-1       
    return (t,X,Y)

def sirs_gsp(n, r0, g = 1, rho = 1, ii = 1, tmax = 10, seed  = 0, debug = False):
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
        ii   : Initial number of infected nodes
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
    t = [0.]
    X = {0.:float(n-ii)}
    Y = {0.:float(ii)}
    random.seed(seed)
    # Check the two exit conditions, extinction and t < tmax
    while Y[t[-1]] and t[-1] < tmax:
        # Calculate transition rates
        rate_SI = g*r0*X[t[-1]]*Y[t[-1]]/n
        rate_IR = g*Y[t[-1]]
        rate_RS = g*rho*(n - X[t[-1]] - Y[t[-1]])
        rate_tot = rate_SI + rate_IR + rate_RS
        if debug: print rate_tot, rate_SI, rate_IR, rate_RS, \
                        g, rho, N - X[t[-1]] - Y[t[-1]]
        # Calculate the time until the next event
        dt = -np.log(random.random())/rate_tot
        t.append(t[-1]+dt)
        # Choose which event happens next
        r = random.random()
        if (r <= rate_SI/rate_tot):
            X[t[-1]] = X[t[-2]]-1
            Y[t[-1]] = Y[t[-2]]+1
        elif rate_SI/rate_tot < r < 1 - rate_RS/rate_tot:
            X[t[-1]] = X[t[-2]]
            Y[t[-1]] = Y[t[-2]]-1
        elif r > 1 - rate_RS/rate_tot:
            X[t[-1]] = X[t[-2]]+1
            Y[t[-1]] = Y[t[-2]]
    return (t,X,Y)

###----------------------------------------------------------------------------
# Functions for calculating groups of trajectories
###----------------------------------------------------------------------------

def sirs_group(n, r0, g = 1, rho = 1, ii = 1, tmax = 10, seed  = 0, nruns = 10):
    """
    Generate a group of trajectories using the Susceptible-Infected-
    Recovered-Susceptible (SIRS) model of disease dynamics.  Each 
    trajectory is calculated in series using sirs_gsp().
    Inputs:
        n    : Population size
        r0   : Epidemiological parameter; this is used to calculate
               the per-contact rate of transmission beta = g*r0/n
        g    : Recovery rate (gamma)
        rho  : Waning immunity rate (rho)
        ii   : Initial number of infected nodes
        tmax : Maximum total length of real time for which the 
               simulation runs
        seed : Seed for random.random()
        nruns: Number of trajectories to simulate in series
    Outputs:
        t   : Times at which observations are made; a list of lists,
              in which t[i] are the times at which the observations are
              made in the ith trajectory
        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
              {time from t[i]:S}, where X[i] is the ith trajectory and
              is indexed by the times in t[i]
        Y   : Numbers of Infecteds vs. time; a list of dictionaries 
              {time from t[i]:I}, where Y[i] is the ith trajectory and
              is indexed by the times in t[i]
    """
    # Lists of observation times
    t_traj = []
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = []
    # Lists of dictionaries, numbers of Infecteds
    Y_traj = []
    # Simulate trajectories in series
    for i in range(nruns):
        (t_sim,X_sim,Y_sim) = sirs_gsp(n, r0, g, rho, ii, tmax, seed)
        # Different random seed for each trajectory
        seed += 1
        t_traj.append(t_sim)
        X_traj.append(X_sim)
        Y_traj.append(Y_sim)
    return t_traj, X_traj, Y_traj

def si_group(n, r0, ii = 1, tmax = 10, seed  = 0, nruns = 10):
    """
    Generate a group of trajectories using the Susceptible-Infected
    model of disease dynamics.  Each trajectory is calculated in 
    series using sis_gsp().
    Inputs:
        n    : Population size
        r0   : Epidemiological parameter; this is used to calculate
               the per-contact rate of transmission beta = g*r0/n
        g    : Recovery rate (gamma)
        ii   : Initial number of infected nodes
        tmax : Maximum total length of real time for which the 
               simulation runs
        seed : Seed for random.random()
        nruns: Number of trajectories to simulate in series
    Outputs:
        t   : Times at which observations are made; a list of lists,
              in which t[i] are the times at which the observations are
              made in the ith trajectory
        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
              {time from t[i]:S}, where X[i] is the ith trajectory and
              is indexed by the times in t[i]
    """
    # Lists of observation times
    t_traj = []
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = []
    # Simulate trajectories in series
    for i in range(nruns):
        (t_sim,X_sim,Y_sim) = si_gsp(n, r0, ii, tmax, seed)
        # Different random seed for each trajectory
        seed += 1
        t_traj.append(t_sim)
        X_traj.append(X_sim)
    return t_traj, X_traj
    
def sis_group(n, r0, g = 1, ii = 1, tmax = 10, seed  = 0, nruns = 10):
    """
    Generate a group of trajectories using the Susceptible-Infected-
    Susceptible model of disease dynamics.  Each trajectory is 
    calculated in series using sis_gsp().
    Inputs:
        n    : Population size
        r0   : Epidemiological parameter; this is used to calculate
               the per-contact rate of transmission beta = g*r0/n
        g    : Recovery rate (gamma)
        ii   : Initial number of infected nodes
        tmax : Maximum total length of real time for which the 
               simulation runs
        seed : Seed for random.random()
        nruns: Number of trajectories to simulate in series
    Outputs:
        t   : Times at which observations are made; a list of lists,
              in which t[i] are the times at which the observations are
              made in the ith trajectory
        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
              {time from t[i]:S}, where X[i] is the ith trajectory and
              is indexed by the times in t[i]
    """
    # Lists of observation times
    t_traj = []
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = []
    # Simulate trajectories in series
    for i in range(nruns):
        (t_sim,X_sim,Y_sim) = sis_gsp(n, r0, g, ii, tmax, seed)
        # Different random seed for each trajectory
        seed += 1
        t_traj.append(t_sim)
        X_traj.append(X_sim)
    return t_traj, X_traj

###----------------------------------------------------------------------------
# Map trajectories onto a grid for easy comparison
###----------------------------------------------------------------------------

def gsp_trajectory_grid(t_traj, X_traj, Y_traj = None, dt = .1):
    """
    Convert the output from any of the three scripts for generating groups 
    of trajectories (sirs_group(), sis_group(), si_group()) into a data 
    format that can be easily manipulated and plotted.  This script
    effectively coarse-grains the trajectories in time, plotting them in
    parallel on a 'grid,' where x-axis is the trajectory index and the
    y-axis is time.
    Inputs:
        t_traj: List of lists of times for a group of trajectories
        X_traj: Numbers of Susceptibles vs. time; a list of dictionaries 
                {time from t[i]:S}, where X[i] is the ith trajectory and
                is indexed by the times in t[i]
        Y_traj: Numbers of Infecteds vs. time; a list of dictionaries 
                {time from t[i]:I}, where Y[i] is the ith trajectory and
                is indexed by the times in t[i]
        dt    : Time step for coarse graining trajectories in time
    Outputs:
        t_grid: List of lists of coarse-grained times for group of 
                trajectories
        X_grid: List of lists of trajectories for Susceptibles
                coarse-grained in time
        Y_grid: List of lists of trajectories for Infecteds, 
                coarse-grained in time, only outputs if Y_traj != None 
    """
    nruns = len(t_traj)
    # Define lists of lists for t, S(t), I(t)
    t_grid = nruns*[[]]
    S_grid = nruns*[[]]
    if Y_traj: I_grid = nruns*[[]]
    max_grid = 0
    # Coarse-grain each trajectory in time
    for i in range(nruns):
        t_grid[i] = scipy.arange(0,t_traj[i][-1]+dt,dt)
        len_grid = len(t_grid[i])
        # Map the real-time time points onto the coarse-grained time steps
        pos_grid = np.searchsorted(t_traj[i],t_grid[i]) 
        S_grid[i] = np.zeros(len_grid)
        if Y_traj: I_grid[i] = np.zeros(len_grid)
        for j in range(len_grid):
            # Fill out the trajectories on the 'grid'
            # One axis of the grid is the different trajectories
            # One axis of the grid is the coarse-grained time
            S_grid[i][j] = X_traj[i][t_traj[i][max(pos_grid[j]-1,0)]]
            if Y_traj: I_grid[i][j] = Y_traj[i][t_traj[i][max(pos_grid[j]-1,0)]]
    if not Y_traj: return t_grid, S_grid
    else: return t_grid, S_grid, I_grid

###----------------------------------------------------------------------------
# Functions for extracting statistics from groups trajectories
###----------------------------------------------------------------------------
    
def avg_profile(t_grid, S_grid, I_grid = None):
    """
    Calculate the average trajectory of a group of trajectories
    Inputs:
            t_grid - coarse-grained time steps
            S_grid - grid of group of trajectories
            I_grid - grid of group of trajectories, not a necessary input
    Outputs:
            t_avg - coarse-grained time steps
            S_avg - mean value of trajectories of number of susceptibles
            S_std - standard deviation of noise about trajectories of number of susceptibles
            I_avg, I_std - mean and std of trajectories of number of infecteds
    """

    # Calculate the average profile
    N_runs = len(S_grid)
    max_grid = np.max([len(i) for i in t_grid])
    dt = t_grid[0][1] - t_grid[0][0] # extract dt
    
    t_avg = np.array(max_grid*[0.])	
    S_avg = np.array(max_grid*[0.])
    S_std = np.array(max_grid*[0.])
    if I_grid: 
        I_avg = np.array(max_grid*[0.])
        I_std = np.array(max_grid*[0.])
        
    for j in range(max_grid):
        if j == 0:
            t_avg[j] = 0
        else:
            t_avg[j] = t_avg[j-1] + dt
        S_avg[j] = np.mean([S_grid[i][j] for i in range(N_runs) if j < len(S_grid[i])])
        S_std[j] = np.std([S_grid[i][j] for i in range(N_runs) if j < len(S_grid[i])])

        if I_grid: 
            I_avg[j] = np.mean([I_grid[i][j] for i in range(N_runs) if j < len(I_grid[i])])
            I_std[j] = np.std([I_grid[i][j] for i in range(N_runs) if j < len(I_grid[i])])
    
    if I_grid:
        return (t_avg,S_avg,I_avg, S_std, I_std)	
    else:
        return (t_avg,S_avg, S_std)


#------------------------------------


def med_profile(t_grid, S_grid, I_grid = None):
    """
    Calculate the median trajectory of a group of trajectories
    Inputs:
            t_grid - coarse-grained time steps
            S_grid - grid of group of trajectories
            I_grid - grid of group of trajectories, not a necessary input
    Outputs:
            t_med - coarse-grained time steps
            S_med - mean value of trajectories of number of susceptibles
            I_med - mean and std of trajectories of number of infecteds
    """

    # Calculate the average profile
    N_runs = len(S_grid)
    max_grid = np.max([len(i) for i in t_grid])
    dt = t_grid[0][1] - t_grid[0][0] # extract dt
    
    t_avg = np.array(max_grid*[0.])	
    S_med = np.array(max_grid*[0.])
    if I_grid: 
        I_med = np.array(max_grid*[0.])
        
    for j in range(max_grid):
        if j == 0:
            t_avg[j] = 0
        else:
            t_avg[j] = t_avg[j-1] + dt
        S_med[j] = np.median([S_grid[i][j] for i in range(N_runs) if j < len(S_grid[i])])

        if I_grid: 
            I_med[j] = np.median([I_grid[i][j] for i in range(N_runs) if j < len(I_grid[i])])
    
    if I_grid:
        return (t_avg,S_med, I_med)	
    else:
        return (t_avg,S_med)
        


#----------------------------------
#  Functions for phase diagram for a group of trajectories
#----------------------------------
def SIRS_gsp_endemic(N, R0s, rhos, g, maxtime, Nruns, se, filename = 'SIRS_gsp_endemic_N.dat'):
    absorb_data = defaultdict(float)
    for R0 in R0s:
        for rho in rhos:
            absorb_data[rho, R0] = 0
            for i in range(Nruns):
                t, X, Y = SIRS_GSP(N, R0, g, rho, N//10, maxtime, se)
                se += 1
                # check to see if the disease has died out...
                if Y[t[-1]] > 1:
                    absorb_data[rho, R0] += 1./Nruns
    f = open(filename, 'w')
    pickle.dump(absorb_data, f)
    f.close()
        


def SIRS_gsp_diagram(N, R0s, rhos, g, maxtime, dt, Nruns, filename = 'SIRS_gsp_diagram_N.dat'):
    data = {}
    for R0 in R0s:
        for rho in rhos:
            t, x, y = SIRS_group(N, R0, g, rho, N//10, maxtime, 0 , Nruns)
            data[rho, R0] = GSP_grid(N, t, x, y, dt)
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()


 
def absorb_diagram(data, pt = False):
    R0s, rhos = ExtractParameters(data)
    absorb_dict = {}
    for R0 in R0s:
        for rho in rhos:
            absorb = 0
            ys = data[R0, rho][-1]
            for y in ys:
                if y[-1] == 0: absorb += 1
            absorb_dict[R0, rho] = 1.*absorb/len(ys)
    if not pt: return absorb_dict
    else:
        #now we plot
        pass

def data_params(idata):
    """
    Given the output from one of the scripts for generating a phase 
    diagram, return two lists of all parameters alphas and R0s.  The 
    input is a single one of the outputs (such as idata or mdata) with
    the form {(alpha, R0):data}
    Inputs:
        idata : dictionary from output of sirs_diagram()
    Outputs:
        alphas: array vector of alpha=rho/gamma
        R0S   : array vector of R0<k>/gamma
    """
    alphas = np.array(sorted(list(set(np.array(idata.keys()).transpose()[0]))))
    R0s = np.array(sorted(list(set(np.array(idata.keys()).transpose()[1]))))
    return alphas, R0s
    
    
#----------------------------------
#  Functions for plotting the analytical results
#----------------------------------

def boundary(R0, g, rho, n, k = 1):
    return (n * (1 - g/R0) * rho/(g + rho) - g/(R0 - g)) - \
            k*np.sqrt(n*(g*((R0 - g)*g*g + (R0*R0 + R0*g - g*g)*rho + 
            2*R0*rho*rho + rho*rho*rho))/(R0*(R0+ rho)*(g + rho)*(g + rho)) +
            (g*g*(g**4 + R0**3*rho + 2*g**3*rho + g*g*rho*rho + rho**4 + 
            R0*R0*(g*g + 2*rho*rho) - R0*(2*g**3 + 3*g*g*rho + 2*g*rho*rho - 
            2*rho**3)))/((R0 - g)*(R0 - g)*rho*rho*(R0 + rho)*(R0 + rho)))
            
def mu(R0, g, rho, n):
    return n * (1 - g/R0) * rho/(g + rho) - g/(R0 - g)
    
def vari(R0, g, rho, n):
    return n*(g*((R0 - g)*g*g + (R0*R0 + R0*g - g*g)*rho + \
            2*R0*rho*rho + rho*rho*rho))/(R0*(R0+ rho)*(g + rho)*(g + rho)) + \
            (g*g*(g**4 + R0**3*rho + 2*g**3*rho + g*g*rho*rho + rho**4 + 
            R0*R0*(g*g + 2*rho*rho) - R0*(2*g**3 + 3*g*g*rho + 2*g*rho*rho - 
            2*rho**3)))/((R0 - g)*(R0 - g)*rho*rho*(R0 + rho)*(R0 + rho))
    