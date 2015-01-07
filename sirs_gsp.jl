using Gadfly

module GSP

using PyCall
@pyimport cPickle as pickle
@pyimport numpy as np
using PyPlot

### Example usage ###
# t, X, Y = sirs_group(1000, 1.1, 1., 10., 50, 100., 0, 100);
# tg, Xg, Yg = gsp_trajectory_grid(t, X, Y, dt = .1);
# tavg, Savg, Svar, Iavg, Ivar = grid_avg(tg, Xg, Yg);

# Things to work on
# *1. Creating whole messes of data - in both series and in parallel
# *2. Visualization, creating heat maps with gadfly, if possible
#   or finally figuring out pyplot
# 3. Benchmark!
# *4. Using map/reduce to parallelize the simulations

###----------------------------------------------------------------------------
# Functions for calculating single trajectories
###----------------------------------------------------------------------------

#    Generate a single trajectory of a Susceptible-Infected (SI)
#    model of disease dynamics.  Exit out of the simulation if the 
#    number of infecteds reaches 0 or if t > tmax.
#    Inputs:
#        n   : Population size
#        r0  : Epidemiological parameter; this is used to calculate
#              the per-contact rate of transmission beta = g*r0/n,
#              where we pick g=1 for SI
#        ii  : Initial number of infected nodes
#        tmax: Maximum total length of real time for which the 
#              simulation runs
#        seed: Seed for random.random
#    Outputs:
#        t   : List of times at which observations are made
#        X   : Dict of number of remaining susceptibles, 
#              indexed by times in t; {time from t:S}
#
function si_gsp(n::Int64, r0, ii::Int64, tmax, seed::Int64 = 0)
    t = [0.]
    X = {0. => float(n-ii)}
    Y = {0. => float(ii)}
    srand(seed)
    # Check the two exit conditions, extinction and t < tmax
    while (X[t[end]] > 0) & (t[end] < tmax)
        # Calculate transition rates
        rate_SI = r0*X[t[end]]*Y[t[end]]/n
        rate_tot = rate_SI
        # Calculate the time until the next event
        dt = -log(rand())/rate_tot
        append!(t, [t[end] + dt])
        # Choose which event happens next
        if rand() <= rate_SI/rate_tot
            X[t[end]] = X[t[end-1]]-1
            Y[t[end]] = Y[t[end-1]]+1
        else
            X[t[end]] = X[t[end-1]]
            Y[t[end]] = Y[t[end-1]]
        end
    end
    t, X, Y
end

#    Generate a single trajectory of a Susceptible-Infected-Susceptible
#    (SIS) model of disease dynamics.  Exit out of the simulation if 
#    the number of infecteds reaches 0 or if t > tmax.
#    Inputs:
#        n   : Population size
#        r0  : Epidemiological parameter; this is used to calculate
#              the per-contact rate of transmission beta = g*r0/n
#        g   : Recovery rate (gamma)
#        ii  : Initial number of infected nodes
#        tmax: Maximum total length of real time for which the 
#              simulation runs
#        seed: Seed for random.random
#    Outputs:
#        t   : List of times at which observations are made
#        X   : Dict of number of remaining Susceptibles, 
#              indexed by times in t; {time from t:S}
#        Y   : Dict of number of Infecteds, indexed by times t;
#              {time from t:I}
#
function sis_gsp(n::Int64, r0, g, 
                 ii::Int64, tmax, seed::Int64 = 0)
    t = [0.]
    X = {0. => float(n-ii)}
    Y = {0. => float(ii)}
    srand(seed)
    # Check the two exit conditions, extinction and t < tmax
    while (Y[t[end]] > 0) & (t[end] < tmax)
        # Calculate transition rates
        rate_SI = g*r0*X[t[end]]*Y[t[end]]/n
        rate_IR = g*Y[t[end]]
        rate_tot = rate_SI + rate_IR
        # Calculate the time until the next event
        dt = -log(rand())/rate_tot
        append!(t, [t[end] + dt])
        # Choose which event happens next
        if rand() <= rate_SI/rate_tot
            X[t[end]] = X[t[end-1]]-1
            Y[t[end]] = Y[t[end-1]]+1
        else
            X[t[end]] = X[t[end-1]]+1
            Y[t[end]] = Y[t[end-1]]-1
        end
    end
    t, X, Y
end

#    Generate a single trajectory of a Susceptible-Infected-Recovered-
#    Susceptible (SIRS) model of disease dynamics.  Exit out of the 
#    simulation if the number of infecteds reaches 0 or if t > tmax.
#    Inputs:
#        n   : Population size
#        r0  : Epidemiological parameter; this is used to calculate
#              the per-contact rate of transmission beta = g*r0/n
#        g   : Recovery rate (gamma)
#        rho : Waning immunity rate (rho)
#        ii  : Initial number of infected nodes
#        tmax: Maximum total length of real time for which the 
#              simulation runs
#        seed: Seed for random.random
#    Outputs:
#        t   : List of times at which observations are made
#        X   : Dict of number of remaining Susceptibles, 
#              indexed by times in t; {time from t:S}
#        Y   : Dict of number of Infecteds, indexed by times t;
#              {time from t:I}
#
function sirs_gsp(n::Int64, r0, g, rho,
                  ii::Int64, tmax, seed::Int64 = 0)
    t = [0.]
    X = {0. => float(n-ii)}
    Y = {0. => float(ii)}
    srand(seed)
    # Check the two exit conditions, extinction and t < tmax
    while (Y[t[end]] > 0) & (t[end] < tmax)
        # Calculate transition rates
        rate_SI = g*r0*X[t[end]]*Y[t[end]]/n
        rate_IR = g*Y[t[end]]
        rate_RS = g*rho*(n - X[t[end]] - Y[t[end]])
        rate_tot = rate_SI + rate_IR + rate_RS
        # Calculate the time until the next event
        dt = -log(rand())/rate_tot
        append!(t, [t[end] + dt])
        # Choose which event happens next
        r = rand()
        if r <= rate_SI/rate_tot
            X[t[end]] = X[t[end-1]]-1
            Y[t[end]] = Y[t[end-1]]+1
        elseif rate_SI/rate_tot < r < 1 - rate_RS/rate_tot
            X[t[end]] = X[t[end-1]]
            Y[t[end]] = Y[t[end-1]]-1
        else r > 1 - rate_RS/rate_tot
            X[t[end]] = X[t[end-1]]+1
            Y[t[end]] = Y[t[end-1]]
        end
    end
    t, X, Y
end

### Gadfly Plotting Example ###
# y = float([Y[ti] for ti in t]);
# plot(x = t, y = y)

###----------------------------------------------------------------------------
# Functions for calculating groups of trajectories
###----------------------------------------------------------------------------

#    Generate a group of trajectories using the Susceptible-Infected-
#    Recovered-Susceptible (SIRS) model of disease dynamics.  Each 
#    trajectory is calculated in series using sirs_gsp().
#    Inputs:
#        n    : Population size
#        r0   : Epidemiological parameter; this is used to calculate
#               the per-contact rate of transmission beta = g*r0/n
#        g    : Recovery rate (gamma)
#        rho  : Waning immunity rate (rho)
#        ii   : Initial number of infected nodes
#        tmax : Maximum total length of real time for which the 
#               simulation runs
#        seed : Seed for random.random()
#        nruns: Number of trajectories to simulate in series
#    Outputs:
#        t   : Times at which observations are made; a list of lists,
#              in which t[i] are the times at which the observations are
#              made in the ith trajectory
#        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
#              {time from t[i]:S}, where X[i] is the ith trajectory and
#              is indexed by the times in t[i]
#        Y   : Numbers of Infecteds vs. time; a list of dictionaries 
#              {time from t[i]:I}, where Y[i] is the ith trajectory and
#              is indexed by the times in t[i]
#
function sirs_group(n::Int64, r0, g, rho,
                    ii::Int64, tmax, seed::Int64 = 0, nruns::Int64 = 10)
    # Lists of observation times
    t_traj = Array(Array, nruns)
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = Array(Dict, nruns)
    # Lists of dictionaries, numbers of Infecteds
    Y_traj = Array(Dict, nruns)
    # Simulate trajectories in series
    for i in 1:nruns
        t_sim, X_sim, Y_sim = sirs_gsp(n, r0, g, rho, ii, tmax, seed)
        # Different random seed for each trajectory
        seed += 1
        t_traj[i] = t_sim
        X_traj[i] = X_sim
        Y_traj[i] = Y_sim
    end
    t_traj, X_traj, Y_traj
end

#    Generate a group of trajectories using the Susceptible-Infected
#    model of disease dynamics.  Each trajectory is calculated in 
#    series using sis_gsp().
#    Inputs:
#        n    : Population size
#        r0   : Epidemiological parameter; this is used to calculate
#               the per-contact rate of transmission beta = g*r0/n
#        g    : Recovery rate (gamma)
#        ii   : Initial number of infected nodes
#        tmax : Maximum total length of real time for which the 
#               simulation runs
#        seed : Seed for random.random()
#        nruns: Number of trajectories to simulate in series
#    Outputs:
#        t   : Times at which observations are made; a list of lists,
#              in which t[i] are the times at which the observations are
#              made in the ith trajectory
#        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
#              {time from t[i]:S}, where X[i] is the ith trajectory and
#              is indexed by the times in t[i]
#              
function si_group(n::Int64, r0, 
                  ii::Int64, tmax, seed::Int64 = 0, nruns::Int64 = 10)
    # Lists of observation times
    t_traj = Array(Array, nruns)
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = Array(Dict, nruns)
    # Simulate trajectories in series
    for i in 1:nruns
        t_sim, X_sim, Y_sim = si_gsp(n, r0, ii, tmax, seed)
        # Different random seed for each trajectory
        seed += 1
        t_traj[i] = t_sim
        X_traj[i] = X_sim
    end
    t_traj, X_traj
end

#    Generate a group of trajectories using the Susceptible-Infected-
#    Susceptible model of disease dynamics.  Each trajectory is 
#    calculated in series using sis_gsp().
#    Inputs:
#        n    : Population size
#        r0   : Epidemiological parameter; this is used to calculate
#               the per-contact rate of transmission beta = g*r0/n
#        g    : Recovery rate (gamma)
#        ii   : Initial number of infected nodes
#        tmax : Maximum total length of real time for which the 
#               simulation runs
#        seed : Seed for random.random()
#        nruns: Number of trajectories to simulate in series
#    Outputs:
#        t   : Times at which observations are made; a list of lists,
#              in which t[i] are the times at which the observations are
#              made in the ith trajectory
#        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
#              {time from t[i]:S}, where X[i] is the ith trajectory and
#              is indexed by the times in t[i]
#              
function sis_group(n::Int64, r0, g,
                  ii::Int64, tmax, seed::Int64 = 0, nruns::Int64 = 10)
    # Lists of observation times
    t_traj = Array(Array, nruns)
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = Array(Dict, nruns)
    # Simulate trajectories in series
    for i in 1:nruns
        t_sim, X_sim, Y_sim = sis_gsp(n, r0, g, ii, tmax, seed)    
        seed += 1
        t_traj[i] = t_sim
        X_traj[i] = X_sim
    end
    t_traj, X_traj
end

###----------------------------------------------------------------------------
# Map trajectories onto a grid for easy comparison
###----------------------------------------------------------------------------

#    Convert the output from any of the three scripts for generating groups 
#    of trajectories (sirs_group(), sis_group(), si_group()) into a data 
#    format that can be easily manipulated and plotted.  This script
#    effectively coarse-grains the trajectories in time, plotting them in
#    parallel on a 'grid,' where x-axis is the trajectory index and the
#    y-axis is time.
#    Inputs:
#        t_traj: List of lists of times for a group of trajectories
#        X_traj: Numbers of Susceptibles vs. time; a list of dictionaries 
#                {time from t[i]:S}, where X[i] is the ith trajectory and
#                is indexed by the times in t[i]
#        Y_traj: Numbers of Infecteds vs. time; a list of dictionaries 
#                {time from t[i]:I}, where Y[i] is the ith trajectory and
#                is indexed by the times in t[i]
#        dt    : Time step for coarse graining trajectories in time
#    Outputs:
#        t_grid: List of lists of coarse-grained times for group of 
#                trajectories
#        X_grid: List of lists of trajectories for Susceptibles
#                coarse-grained in time
#        Y_grid: List of lists of trajectories for Infecteds, 
#                coarse-grained in time, only outputs if Y_traj != None 
#
function gsp_trajectory_grid(t_traj, X_traj, Y_traj = None; dt = .1)
    nruns = length(t_traj)
    # Define lists of lists for t, S(t), I(t)
    t_grid = Array(Array, nruns)
    S_grid = Array(Array, nruns)
    if Y_traj != None
        I_grid = Array(Array, nruns)
    end
    max_grid = 0
    for i in 1:nruns
        t_grid[i] = [0:dt:t_traj[i][end]+dt]
        len_grid = length(t_grid[i])
        # Map the real-time time points onto the coarse-grained time steps
        pos_grid = [searchsortedfirst(t_traj[i], j) for j in t_grid[i]]
        S_grid[i] = Array(Float64, len_grid)
        if Y_traj != None
            I_grid[i] = Array(Float64, len_grid)
        end
        for j in 1:len_grid
            S_grid[i][j] = X_traj[i][t_traj[i][max(pos_grid[j]-1,1)]]
            if Y_traj != None
                I_grid[i][j] = Y_traj[i][t_traj[i][max(pos_grid[j]-1,1)]]
            end
            # Fill out the trajectories on the 'grid'
            # One axis of the grid is the different trajectories
            # One axis of the grid is the coarse-grained time
        end
    end
    if Y_traj != None
        t_grid, S_grid, I_grid
    else
        t_grid, S_grid
    end
end

###----------------------------------------------------------------------------
# Functions for extracting statistics from groups trajectories
###----------------------------------------------------------------------------

#    Calculate the average and standard deviation of a group of 
#    trajectories.  The average is computed across all trajectories,
#    not over time. Need to coarse-grain in time using 
#    gsp_trajectory_grid() first.
#    Inputs:
#        t_grid : Coarse-grained time steps
#        S_grid : Number of Susceptibles for a group of trajectories,
#                 coarse-grained in time
#        I_grid : Number of Infecteds for a group of trajectories,
#                 coarse-grained in time
#    Outputs:
#        t_avg  : coarse-grained time steps
#        S_avg  : Mean value of number of Susceptibles, averaged
#                 across all trajectories 
#        S_std  : Standard deviation of the number of Susceptibles,
#                 averaged across all trajectories
#        I_avg  : Mean value of number of Iusceptibles, averaged
#                 across all trajectories. This is computed only
#                 if I_grid != None.
#        I_std  : Standard deviation of the number of Infecteds,
#                 averaged across all trajectories. This is computed 
#                 only if I_grid != None.
#
function grid_avg(t_grid, S_grid, I_grid = None)
    nruns = length(S_grid)
    max_grid = maximum([length(i) for i in t_grid])
    # Extract time step size dt from t_grid
    dt = t_grid[1][2] - t_grid[1][1]
    # Initialize arrays
    t_avg = Array(Float64, max_grid)
    S_avg = Array(Float64, max_grid)
    S_std = Array(Float64, max_grid)
    if I_grid != None
        I_avg = Array(Float64, max_grid)
        I_std = Array(Float64, max_grid)
    end
    # Compute statistics across all trajectories
    for j in 1:max_grid
        if j == 1
            t_avg[j] = 0
        else
            t_avg[j] = t_avg[j-1] + dt 
        end
        # Ignore all data points where the trajectories have died out
        sdataj = [if j <= length(S_grid[i])
                    S_grid[i][j]
                  else
                    -1
                  end 
                  for i in 1:nruns]
        sdataj = float(sdataj[sdataj.!=-1])
        S_avg[j] = mean(sdataj)
        S_std[j] = std(sdataj, corrected = false)
        # print out 0 if only 1 trajectory remains
        if I_grid != None
            idataj = [if j <= length(I_grid[i])
                        I_grid[i][j]
                      else
                        -1
                      end 
                      for i in 1:nruns]
            idataj = float(idataj[idataj.!=-1])
            I_avg[j] = mean(idataj)
            I_std[j] = std(idataj, corrected = false)
        end
    end
    if I_grid != None
        t_avg, S_avg, S_std, I_avg, I_std
    else
        t_avg, S_avg, S_std
    end
end

#    Calculate the median of a group of trajectories, computed across 
#    all trajectories, not over time. Need to coarse-grain in time 
#    using gsp_trajectory_grid() first.
#    Inputs:
#        t_grid : Coarse-grained time steps
#        S_grid : Number of Susceptibles for a group of trajectories,
#                 coarse-grained in time
#        I_grid : Number of Infecteds for a group of trajectories,
#                 coarse-grained in time
#    Outputs:
#        t_avg  : coarse-grained time steps
#        S_med  : Median value of number of Susceptibles, computed
#                 across all trajectories 
#        I_med  : Median value of number of Iusceptibles, computed
#                 across all trajectories. This is computed only
#                 if I_grid != None.
#
function grid_med(t_grid, S_grid, I_grid = None)
    nruns = length(S_grid)
    max_grid = maximum([length(i) for i in t_grid])
    # Extract time step size dt from t_grid
    dt = t_grid[1][2] - t_grid[1][1]
    # Initialize arrays
    t_avg = Array(Float64, max_grid)
    S_med = Array(Float64, max_grid)
    if I_grid != None
        I_med = Array(Float64, max_grid)
    end
    # Compute statistics across all trajectories
    for j in 1:max_grid
        if j == 1
            t_avg[j] = 0
        else
            t_avg[j] = t_avg[j-1] + dt 
        end
        # Ignore all data points where the trajectories have died out
        sdataj = [if j <= length(S_grid[i])
                    S_grid[i][j]
                  else
                    -1
                  end 
                  for i in 1:nruns]
        sdataj = float(sdataj[sdataj.!=-1])
        S_med[j] = median(sdataj)
        # print out 0 if only 1 trajectory remains
        if I_grid != None
            idataj = [if j <= length(I_grid[i])
                        I_grid[i][j]
                      else
                        -1
                      end 
                      for i in 1:nruns]
            idataj = float(idataj[idataj.!=-1])
            I_med[j] = median(idataj)
        end
    end
    if I_grid != None
        t_avg, S_med, I_med
    else
        t_avg, S_med
    end
end

###----------------------------------------------------------------------------
#  Functions for phase diagram for a group of trajectories
###----------------------------------------------------------------------------

#    Produce a group of SIRS trajectories for each given parameter 
#    combination.  Thus, this produces a phase diagram that can
#    be used to measure the persistence of the endemic phase,
#    statistics of trajectories, and more.
#    Input:
#        n      : Population size
#        r0s    : Array/list of R0 values in parameter space
#        alphas : Array/list of alpha values in parameter space
#        g      : Recovery rate (gamma)
#        maxtime: Maximum total length of real time for which the 
#                 simulation runs
#        dt     : Time step
#        seed   : Initial seed for random.random()
#        nruns  : Number of trajectories to simulate in series
#                 for each set of parameters
#        fname  : If filename != None, write the output out to the
#                 named location.  Else return the output.
#    Output:
#        data   : This is a dictionary containing the full coarse-
#                 grained trajectory data for a group of trajectories
#                 calculated for each parameter combination.
#                 {[alpha, r0] : coarse-grained trajectory group}
#                 If fname != None, write out to file at given location;
#                 otherwise, return data.
#
function sirs_diagram(n, r0s, alphas, g, maxtime, dt, nruns, 
                 fname = "SIRS_gsp_diagram_N.dat")
    data = Dict()
    for r0 in r0s
        for alpha in alphas
            #println(alpha, " ", r0)
            t, x, y = sirs_group(n, r0, g, alpha, div(n,10), maxtime, 0 , nruns);
            data[alpha, r0] = gsp_trajectory_grid(t, x, y, dt=dt);
        end
    end
    if fname != None
        #println("writing out now")
        f = open(fname, "w");
        pickle.dump(PyDict(data), f);
        #println("saving")
        close(f)
    else
        data
    end
end

#    Create a phase diagram of the fraction of extinct (absorbed)
#    trajectories.
#    Inputs:
#        data : Output from sirs_diagram() above
#    Outputs:
#        absorb_dict : 
#               Fraction of absorbed trajectories at each
#               parameter combination: {[alpha, R0] : fraction absorbed}
#
function absorb_diagram(data)
    alphas, r0s = data_params(data)
    absorb_dict = Dict()
    for r0 in r0s
        for alpha in alphas
            absorb = 0
            ys = data[alpha, r0][end]
            for y in ys
                if y[end] == 0
                    absorb += 1
                end
            end
            absorb_dict[alpha, r0] = 1.*absorb/length(ys)
        end
    end
    absorb_dict
end

#    Use pcolormesh from matplotlib to create a visualization of the 
#    SIRS phase diagram in the form of a heat map.
#    Input:  
#        data  : Dictionary of the form {(alpha, R0}:value}, where
#                the value represents <I*>, <S*>, etc.  This dictionary
#                is produced by avg_idata(), fluct(), or frac_fluct()
#        alphas: Phase diagram parameters on x-axis; rho/gamma
#        R0s   : Phase diagram parameters on y-axis; beta<k>/gamma
#                These can be obtained with data_params()
#        p     : Create a plot of the data if p==True
#        logx  : Plot with a logarithmic x-axis (alphas logarithmic)
#        ret   : If ret==True, return the values used to create the
#                heat map, so that they may be manipulated in the 
#                interpreter or elsewhere; defaults to False
#    Output:
#        Creates a matplotlib heatmap if p==True
#        (X, Y, C): These are three arrays that pcolormesh() takes as
#                arguments.  These are returned only if ret==True
#
function colormap(data, alphas, r0s,
                  p = true, logx = true, ret = false)
    X, Y = np.meshgrid(alphas, r0s);
    C = Float64[data[alpha, r0] for alpha in alphas, r0 in r0s];
    if p
        plt.figure()
        plt.pcolormesh(X, Y, C)
        plt.xlabel("rho/gamma")
        if logx
            plt.semilogx()
        end
        plt.ylabel("R_0")
        plt.colorbar()
        plt.show()
    end
    if ret
        X, Y, C
    end
end

#    Given the output from one of the scripts for generating a phase 
#    diagram, return two lists of all parameters alphas and R0s.  The 
#    input is a single one of the outputs (such as idata or mdata) with
#    the form {(alpha, R0):data}
#    Inputs:
#        idata : dictionary from output of sirs_diagram()
#    Outputs:
#        alphas: array vector of alpha=rho/gamma
#        R0S   : array vector of R0<k>/gamma
#
function data_params(data)
    alphas = Float64[]
    R0s = Float64[]
    for k in keys(data)
        push!(alphas, k[1]);
        push!(R0s, k[2]);
    end
    alphas = sort([i for i in Set(alphas)]);
    R0s = sort([i for i in Set(R0s)]);
    return alphas, R0s
end


###----------------------------------------------------------------------------
# End GSP module
###----------------------------------------------------------------------------
export si_gsp, sis_gsp, sirs_gsp
export si_group, sis_group, sirs_group
export gsp_trajectory_grid, grid_avg, sirs_diagram
export data_params, absorb_diagram, colormap
end


###----------------------------------------------------------------------------
# Begin parallel module
###----------------------------------------------------------------------------

### Example usage
### This is how you need to open Julia, n=number of cores, for multithreading
# julia -p n
### This is how you need to call before using, otherwise horrible errors
# @everywhere include("sirs_gsp.jl")
# using GSP_par
# using GSP
### For benchmarking, to show that this really is faster
### In series:
# n = 1000; r0 = 2; g = 1.; rho = 10; ii = 10; tmax = 5; seed = 0; nruns = 1000;
### tic(); t, X, Y = sirs_group(n, r0, g, rho, ii, tmax, seed, nruns); toc()
### In parallel:
# tic(); t, X, Y = sirs_group_par(n, r0, g, rho, ii, tmax, seed, nruns); toc()
### Can easily compare the calculated average trajectories, find same results
### (except that the parallel code is faster!)
# tg, Xg, Yg = gsp_trajectory_grid(t, X, Y);
# tpg, Xpg, Ypg = gsp_trajectory_grid(tp, Xp, Yp);
# ta, Xa, Xs, Ya, Ys = grid_avg(tg, Xg, Yg);
# tpa, Xpa, Xps, Ypa, Yps = grid_avg(tpg, Xpg, Ypg);
# [ta tpa]
# [Xa Xpa]
# [Ya Ypa]

module GSP_par

using GSP

using PyCall
@pyimport cPickle as pickle
@pyimport numpy as np

#    Generate a group of trajectories using the Susceptible-Infected-
#    Recovered-Susceptible (SIRS) model of disease dynamics.  Each 
#    trajectory is calculated in series using sirs_gsp().
#    Inputs:
#        n    : Population size
#        r0   : Epidemiological parameter; this is used to calculate
#               the per-contact rate of transmission beta = g*r0/n
#        g    : Recovery rate (gamma)
#        rho  : Waning immunity rate (rho)
#        ii   : Initial number of infected nodes
#        tmax : Maximum total length of real time for which the 
#               simulation runs
#        seed : Seed for random.random()
#        nruns: Number of trajectories to simulate in series
#    Outputs:
#        t   : Times at which observations are made; a list of lists,
#              in which t[i] are the times at which the observations are
#              made in the ith trajectory
#        X   : Numbers of Susceptibles vs. time; a list of dictionaries 
#              {time from t[i]:S}, where X[i] is the ith trajectory and
#              is indexed by the times in t[i]
#        Y   : Numbers of Infecteds vs. time; a list of dictionaries 
#              {time from t[i]:I}, where Y[i] is the ith trajectory and
#              is indexed by the times in t[i]
#
function sirs_group_par(n::Int64, r0, g, rho,
                    ii::Int64, tmax, seed::Int64 = 0, nruns::Int64 = 10)
    # Lists of observation times
    t_traj = Array(Array, nruns)
    # Lists of dictionaries, numbers of Susceptibles
    X_traj = Array(Dict, nruns)
    # Lists of dictionaries, numbers of Infecteds
    Y_traj = Array(Dict, nruns)
    # Create array of parameters for parallel simulations
    work=Array(Any,nruns)
    for i in 1:nruns
        work[i]=(n, r0, g, rho, ii, tmax, seed)
        seed += 1;
    end
    # Simulate trajectories in parallel with pmap
    holder=pmap(work) do package
        apply(sirs_gsp, package)
    end
    #Reduce the pmapped data to usable form
    for i in 1:nruns
        t_traj[i], X_traj[i], Y_traj[i] = holder[i]
    end
    t_traj, X_traj, Y_traj
end

###----------------------------------------------------------------------------
#  Functions for phase diagram for a group of trajectories
###----------------------------------------------------------------------------

#    Produce a group of SIRS trajectories for each given parameter 
#    combination.  Thus, this produces a phase diagram that can
#    be used to measure the persistence of the endemic phase,
#    statistics of trajectories, and more.
#    Input:
#        n      : Population size
#        r0s    : Array/list of R0 values in parameter space
#        alphas : Array/list of alpha values in parameter space
#        g      : Recovery rate (gamma)
#        maxtime: Maximum total length of real time for which the 
#                 simulation runs
#        dt     : Time step
#        seed   : Initial seed for random.random()
#        nruns  : Number of trajectories to simulate in series
#                 for each set of parameters
#        fname  : If filename != None, write the output out to the
#                 named location.  Else return the output.
#    Output:
#        data   : This is a dictionary containing the full coarse-
#                 grained trajectory data for a group of trajectories
#                 calculated for each parameter combination.
#                 {[alpha, r0] : coarse-grained trajectory group}
#                 If fname != None, write out to file at given location;
#                 otherwise, return data.
#
function sirs_diagram(n, r0s, alphas, g, maxtime, dt, nruns, 
                 fname = "SIRS_gsp_diagram_N.dat")
    data = Dict()
    for r0 in r0s
        for alpha in alphas
            #println(alpha, " ", r0)
            t, x, y = sirs_group_par(n, r0, g, alpha, div(n,10), maxtime, 0 , nruns);
            data[alpha, r0] = gsp_trajectory_grid(t, x, y, dt=dt);
        end
    end
    if fname != None
        #println("writing out now")
        f = open(fname, "w");
        pickle.dump(PyDict(data), f);
        #println("saving")
        close(f)
    else
        data
    end
end



###----------------------------------------------------------------------------
# End module
###----------------------------------------------------------------------------
export sirs_group_par, sirs_diagram
end