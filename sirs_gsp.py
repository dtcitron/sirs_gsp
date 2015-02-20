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
#  Updates for later:
#  1. I have also attempted to extend the Gillespie algorithm for SIR-type
#     simulations on complex networks (given a connectivity graph), but
#     this is not yet optimized.
#  2. Is there a simple way to use map() to calculate groups of 
#     trajectories in parallel, rather than in series?
#  3. There is a smarter way of drawing from the exponential distribution? 
#     ziggurat method: internal to Julia and Python

import numpy as np
import scipy
import random
import cPickle as pickle
from collections import defaultdict
import scipy.optimize as opt
import matplotlib.pyplot as plt

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
            if Y_traj: I_grid[i][j]= Y_traj[i][t_traj[i][max(pos_grid[j]-1,0)]]
    if not Y_traj: return t_grid, S_grid
    else: return t_grid, S_grid, I_grid

###----------------------------------------------------------------------------
# Functions for extracting statistics from groups trajectories
###----------------------------------------------------------------------------
    
def grid_avg(t_grid, S_grid, I_grid = None):
    """
    Calculate the average and standard deviation of a group of 
    trajectories.  The average is computed across all trajectories,
    not over time. Need to coarse-grain in time using 
    gsp_trajectory_grid() first.
    Inputs:
        t_grid : Coarse-grained time steps
        S_grid : Number of Susceptibles for a group of trajectories,
                 coarse-grained in time
        I_grid : Number of Infecteds for a group of trajectories,
                 coarse-grained in time
    Outputs:
        t_avg  : coarse-grained time steps
        S_avg  : Mean value of number of Susceptibles, averaged
                 across all trajectories 
        S_std  : Standard deviation of the number of Susceptibles,
                 averaged across all trajectories
        I_avg  : Mean value of number of Iusceptibles, averaged
                 across all trajectories. This is computed only
                 if I_grid != None.
        I_std  : Standard deviation of the number of Infecteds,
                 averaged across all trajectories. This is computed 
                 only if I_grid != None.
    """
    nruns = len(S_grid)
    max_grid = np.max([len(i) for i in t_grid])
    # Extract time step size dt from t_grid
    dt = t_grid[0][1] - t_grid[0][0]
    # Initialize arrays
    t_avg = np.array(max_grid*[0.])	
    S_avg = np.array(max_grid*[0.])
    S_std = np.array(max_grid*[0.])
    if I_grid: 
        I_avg = np.array(max_grid*[0.])
        I_std = np.array(max_grid*[0.])
    # Compute statistics across all trajectories
    for j in range(max_grid):
        if j == 0:
            t_avg[j] = 0
        else:
            t_avg[j] = t_avg[j-1] + dt
        # Ignore all data points where the trajectories have died out
        S_avg[j] = np.mean([S_grid[i][j] for i in range(nruns) \
                            if j < len(S_grid[i])])
        S_std[j] = np.std([S_grid[i][j] for i in range(nruns) \
                            if j < len(S_grid[i])])
        if I_grid: 
            I_avg[j] = np.mean([I_grid[i][j] for i in range(nruns) \
                                if j < len(I_grid[i])])
            I_std[j] = np.std([I_grid[i][j] for i in range(nruns) \
                                if j < len(I_grid[i])])
    if I_grid:
        return (t_avg, S_avg, S_std, I_avg, I_std)	
    else:
        return (t_avg, S_avg, S_std)

def grid_med(t_grid, S_grid, I_grid = None):
    """
    Calculate the median of a group of trajectories, computed across 
    all trajectories, not over time. Need to coarse-grain in time 
    using gsp_trajectory_grid() first.
    Inputs:
        t_grid : Coarse-grained time steps
        S_grid : Number of Susceptibles for a group of trajectories,
                 coarse-grained in time
        I_grid : Number of Infecteds for a group of trajectories,
                 coarse-grained in time
    Outputs:
        t_avg  : coarse-grained time steps
        S_med  : Median value of number of Susceptibles, computed
                 across all trajectories 
        I_med  : Median value of number of Iusceptibles, computed
                 across all trajectories. This is computed only
                 if I_grid != None.
    """
    nruns = len(S_grid)
    max_grid = np.max([len(i) for i in t_grid])
    # Extract time step size dt from t_grid
    dt = t_grid[0][1] - t_grid[0][0]
    # Initialize arrays
    t_avg = np.array(max_grid*[0.])	
    S_med = np.array(max_grid*[0.])
    if I_grid: 
        I_med = np.array(max_grid*[0.])
    for j in range(max_grid):
        if j == 0:
            t_avg[j] = 0
        else:
            t_avg[j] = t_avg[j-1] + dt
        # Ignore all data points where the trajectories have died out
        S_med[j] = np.median([S_grid[i][j] for i in range(nruns) \
                            if j < len(S_grid[i])])
        if I_grid: 
            I_med[j] = np.median([I_grid[i][j] for i in range(nruns) \
                            if j < len(I_grid[i])])
    if I_grid:
        return (t_avg,S_med, I_med)	
    else:
        return (t_avg,S_med)
        
###----------------------------------------------------------------------------
#  Functions for phase diagram for a group of trajectories
###----------------------------------------------------------------------------

def sirs_diagram(n, r0s, alphas, g, maxtime, dt, nruns, 
                 fname = 'SIRS_gsp_diagram_N.dat'):
    """
    Produce a group of SIRS trajectories for each given parameter 
    combination.  Thus, this produces a phase diagram that can
    be used to measure the persistence of the endemic phase,
    statistics of trajectories, and more.
    Input:
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
    Output:
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
            t, x, y = sirs_group(n, r0, g, alpha, n//10, maxtime, 0 , nruns)
            data[alpha, r0] = gsp_trajectory_grid(t, x, y, dt)
    if fname != None:
        f = open(fname, 'w')
        pickle.dump(data, f)
        f.close()
    else:
        return data

def absorb_diagram(data):
    """
    Create a phase diagram of the fraction of extinct (absorbed)
    trajectories.
    Inputs:
        data : Output from sirs_diagram() above
    Outputs:
        absorb_dict : 
               Fraction of absorbed trajectories at each
               parameter combination: {[alpha, R0] : fraction absorbed}
    """
    alphas, r0s = data_params(data)
    absorb_dict = {}
    for r0 in r0s:
        for alpha in alphas:
            absorb = 0
            ys = data[alpha, r0][-1]
            for y in ys:
                if y[-1] == 0: absorb += 1
            absorb_dict[alpha, r0] = 1.*absorb/len(ys)
    return absorb_dict
    
def snr_diagram(data, t_index = None):
    """
    Create a phase diagram of the signal to noise ratio of non-extinct
    (still going) trajectories.  Specifically, find the SNR of the 
    quasi-stationary distribution of the number of infecteds at a given
    time index
    Inputs:
        data : Output from sirs_diagram() above
        t_index: Time at which we observe the QSD.  Defaults to None
    Outputs:
        absorb_dict : 
               Fraction of absorbed trajectories at each
               parameter combination: {[alpha, R0] : fraction absorbed}
    """
    alphas, r0s = data_params(data)
    snr_dict = {}
    for r0 in r0s:
        for alpha in alphas:
            # find all infecteds at t_index
            if t_index == None:
                infecteds = np.array([y[-1] for y in data[alpha, r0][-1]])
            else:
                infecteds = np.array([y[t_index] for y in data[alpha, r0][-1]\
                            if t_index < len(y)])
            # condition on the trajectory not dying out
            qsdi = infecteds[np.where(infecteds > 0)]
            if len(qsdi)>1:
                m = np.mean(qsdi)
                s = np.std(qsdi)
                snr_dict[alpha, r0] = m/s
            else:
                snr_dict[alpha, r0] = 0
    return snr_dict

def colormap(data, alphas, R0s,
             p = True, logx = True, ret = False):
    """
    Use pcolormesh from matplotlib to create a visualization of the 
    SIRS phase diagram in the form of a heat map.
    Input:  
        data  : Dictionary of the form {(alpha, R0}:value}, where
                the value represents <I*>, <S*>, etc.  This dictionary
                is produced by avg_idata(), fluct(), or frac_fluct()
        alphas: Phase diagram parameters on x-axis; rho/gamma
        R0s   : Phase diagram parameters on y-axis; beta<k>/gamma
                These can be obtained with data_params()
        p     : Create a plot of the data if p==True
        logx  : Plot with a logarithmic x-axis (alphas logarithmic)
        ret   : If ret==True, return the values used to create the
                heat map, so that they may be manipulated in the 
                interpreter or elsewhere; defaults to False
    Output:
        Creates a matplotlib heatmap if p==True
        (X, Y, C): These are three arrays that pcolormesh() takes as
                arguments.  These are returned only if ret==True
    """
    X, Y = np.meshgrid(alphas, R0s)
    C = np.array([[data[rho_, R0] for rho_ in alphas] for R0 in R0s])
    if p:
        plt.figure()
        plt.pcolormesh(X, Y, C)
        plt.xlabel('rho/gamma')
        if logx: 
            plt.semilogx()
        plt.ylabel('R_0')
        plt.colorbar()
        plt.show()
    if ret: return X, Y, C

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
    
def sirs_diagram_endemic(n, r0s, alphas, g, maxtime, seed, nruns,
                         fname = 'SIRS_gsp_endemic_N.dat'):
    """
    Produce the phase diagram of where in model parameter space the
    endemic disease is sustained up to maxtime.  For each combination
    of parameters, we count the fraction of SIRS trajectories that 
    survive up to maxtime.  Initially infect 10% of the population.
    Inputs:
        n      : Population size
        r0s    : Array/list of R0 values in parameter space
        alphas : Array/list of alpha values in parameter space
        g      : Recovery rate (gamma)
        maxtime: Maximum total length of real time for which the 
                 simulation runs
        seed   : Initial seed for random.random()
        nruns  : Number of trajectories to simulate in series
                 for each set of parameters
        fname  : If fname != None, write the output out to the
                 named location.  Else return the output.
    Output:
        absorb_data : This is a dictionary containing the fraction of
                 trajectories that survive for each combination of 
                 parameters. {[alpha, R0] : fraction surviving}.
                 absorb_data is either written to a file or if 
                 fname == None we return absorb_data.
    """
    absorb_data = defaultdict(float)
    for r0 in r0s:
        for alpha in alphas:
            absorb_data[alpha, r0] = 0
            for i in range(nruns):
                t, X, Y = sirs_gsp(n, r0, g, alpha, n//10, maxtime, seed)
                seed += 1
                # check to see if the disease has died out...
                if Y[t[-1]] > 0:
                    absorb_data[alpha, r0] += 1./nruns
    if fname != None:
        f = open(fname, 'w')
        pickle.dump(absorb_data, f)
        f.close()
    else:
        return absorb_data