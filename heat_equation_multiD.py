#Make sure running with python3.7 and tensorflow 1.5.4(not tf2)
#tensorflow_probability 0.7.0
#%% import needed packages

import DGM
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt
import math
from heat_plot import *
from time import time

#%% Parameters 
k=1
eps = 1e-5
dim = 2
l1_loss_scale = 1
l2_loss_scale = 1
l3_loss_scale = 1

# Solution parameters (domain on which to solve PDE)
t_low = 0 + eps   # time lower bound
t_high = 2
#Always assuming square domain...
x_low = -1
x_high = 1

# neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training parameters
sampling_stages  = 100   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 100
nSim_bound = 20#200 for each boundary
nSim_initial = 20
S_multiplier  = 1.5   # multiplier for oversampling i.e. draw S from [S_low, S_high * S_multiplier]

# Plot options
n_plot = 41  # Points on plot grid for each dimension

# Save options
saveOutput = True
saveName   = 'HeatEquation_multi_dim'
saveFigure = True
figureName = 'HeatEquation_multi_dim.png'

#%% Black-Scholes European call price

'''Discussion of heat 1-d equation:
    u_t = ku_xx where k is constant
    In general u_t = (spatial laplacian)[u]

    Two conditions:
        initial: u(x,0) for x in domain
        boundary: u(0,t),u(1,t) for t in time

    Fundamental(One initial heat source) solution:
        u(x,t) = 1/(sqrt(4 pi k t)) e^(-x^2/4kt)
        on R x (0,infty)

    Setup:
        Spatial Domain: [0,1]
        Time Domain: [0,1]
        Initial condition: u(0,0) = 1
        No boundary condition

'''

#evaluating blackScholes
def HeatCall(x,t):
    exp = 0
    frac = 0
    if t == 0:
        exp = np.prod(np.exp(-3*x**2))
        frac = np.sqrt(1/(4 * math.pi))**dim
    else:
        exp = np.prod(np.exp(-x**2/(4*k*t)),axis=1)
        #add eps for t= 0
        frac = np.sqrt(1/(4 * math.pi * k *t + eps))**dim

    return frac*exp
    

    

#%% Sampling function - randomly sample time-space pairs 

def sampler(nSim_interior, nSim_boundary, nSim_terminal):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler #1: domain interior
    t_interior = np.random.uniform(low=t_low, high=t_high, size=[nSim_interior, 1])
    S_interior = np.random.uniform(low=x_low, high=x_high, size=[nSim_interior, dim])

    # Sampler #2: spatial boundary
    t_bound = np.random.uniform(low=t_low,high=t_high,size=[2*dim*nSim_boundary,1])
    S_bound = np.array([])
    for i in range(dim):
        left_bound = np.random.uniform(low=x_low,high=x_high,size=[nSim_boundary,dim-1])
        left_bound = np.insert(left_bound,i,-1.0,axis=1)
        right_bound = np.random.uniform(low=x_low,high=x_high,size=[nSim_boundary,dim-1])
        right_bound = np.insert(right_bound,i,1.0,axis=1)
        if dim == 1:
            left_bound = np.zeros((nSim_boundary,1))-1
            right_bound = np.zeros((nSim_boundary,1))+1
        dim_bound = np.concatenate((left_bound,right_bound),axis=0)
        S_bound = np.append(S_bound,dim_bound)
    S_bound = np.reshape(S_bound,(2*dim*nSim_boundary,dim))
    #print(S_bound)
    
    # Sampler #3: initial/terminal condition
    t_init = np.zeros((nSim_terminal, 1))
    #Change sampling strategy to sample in areas that matter more
    #Should bound domain somehow
    S_init = np.random.uniform(low=x_low, high=x_high, size = [nSim_terminal,dim])

    return t_interior, S_interior, t_bound, S_bound, t_init, S_init

#%% Loss function for Fokker-Planck equation

def loss(model, t_interior, S_interior, t_bound, S_bound, t_terminal, S_terminal):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        S_interior: sampled space points in the interior of the function's domain
        t_terminal: sampled time points at terminal point (vector of terminal times)
        S_terminal: sampled space points at terminal time
    ''' 
    
    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    #Is this predicted value?
    #No clue how stuff in here is typechecking...
    V = model(t_interior, S_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_s = tf.gradients(V, S_interior)[0]
    #This is the hessian
    V_ss = tf.gradients(V_s, S_interior)[0]
    #laplacian = tf.linalg.trace(V_ss)
    laplacian = V_ss
    #Not sure how indexing will work
    #for i in range(dim):
    #    V_s = tf.gradients(V,S_interior)[i]
    #    V_ss = tf.gradients(V_s,S_interior)[i]
    #    laplacian = laplacian + V_ss
    #This is the pde to model
    diff_V = V_t - k*laplacian

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V)) 
    
    # Loss term #2: boundary condition
    #Will want 0 at boundary term
    fitted_bound = model(t_bound,S_bound)
    L2 = tf.reduce_mean(tf.square(fitted_bound))
    
    # Loss term #3: initial/terminal condition
    #Target is boundary function
    #Try to avoid floating point equality
    #Placeholder is kind of like \cdot(precomposition)
    #Currently initial condition is tight gaussian heat spike
    #No clue how these conditions working in 2d: works symbolically but not in my other mental model? Magical
    gauss = lambda x : np.exp(-10*(x)**2)
    tf_gauss = tf.py_function(func=gauss,inp=[S_terminal],Tout=tf.float32)
    
    target_payoff = tf.math.reduce_prod(tf.math.exp(tf.math.scalar_mul(-3.0,tf.math.square(S_terminal))),axis=1)
    target_payoff = tf_gauss
    #Must make sure model is treating inputs correctly
    fitted_payoff = model(t_terminal, S_terminal)
    
    L3 = tf.reduce_mean( tf.square(fitted_payoff - target_payoff) )

    return l1_loss_scale*L1, l2_loss_scale*L2, l3_loss_scale*L3

#%% Set up network

#Very convenient
# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, dim)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
S_interior_tnsr = tf.placeholder(tf.float32, [None,dim])
t_init_tnsr = tf.placeholder(tf.float32, [None,1])
S_init_tnsr = tf.placeholder(tf.float32, [None,dim])
t_bound_tnsr = tf.placeholder(tf.float32,[None,1])
S_bound_tnsr = tf.placeholder(tf.float32,[None,dim])


# loss 
L1_tnsr, L2_tnsr, L3_tnsr = loss(model, t_interior_tnsr, S_interior_tnsr, t_bound_tnsr, S_bound_tnsr, t_init_tnsr, S_init_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr

# option value function
V = model(t_interior_tnsr, S_interior_tnsr)

# set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%% Train network
# for each sampling stage
losses = []
l1_losses = []
l2_losses = []
l3_losses = []
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    t_interior, S_interior, t_bound, S_bound, t_terminal, S_terminal = sampler(nSim_interior, nSim_bound, nSim_initial)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L2,L3,_ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, optimizer],
                                feed_dict = {t_interior_tnsr:t_interior, S_interior_tnsr:S_interior,t_bound_tnsr:t_bound,S_bound_tnsr:S_bound, t_init_tnsr:t_terminal, S_init_tnsr:S_terminal})
    
    print(loss, L1, L2, L3, i)
    losses.append(loss)
    l1_losses.append(L1)
    l2_losses.append(L2)
    l3_losses.append(L3)

plot_loss(losses,"heat_multiDim/heat_multi_total_loss")
plot_loss(l1_losses,"heat_multiDim/heat_multi_l1_loss")
plot_loss(l1_losses,"heat_multiDim/heat_multi_l2_loss")
plot_loss(l3_losses,"heat_multiDim/heat_multi_l3_loss")

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)

#%% Plot results
#Comparison
if dim == 1:
    print("1 dimension")
        # LaTeX rendering for text in plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # figure options
    plt.figure()
    plt.figure(figsize = (12,10))

    # time values at which to examine density
    valueTimes = [t_low, (2/3)*t_low+(1/3)*t_high, (1/3)*t_low+(2/3)*t_high, t_high]

    # vector of t and S values for plotting
    S_plot = np.linspace(x_low, x_high, n_plot)

    for i, curr_t in enumerate(valueTimes):
        
        # specify subplot
        plt.subplot(2,2,i+1)
        
        # simulate process at current t 
        #Note this is vectorized
        optionValue = HeatCall(np.reshape(S_plot,(-1,1)), curr_t)
        
        # compute normalized density at all x values to plot and current t value
        t_plot = curr_t * np.ones_like(S_plot.reshape(-1,1))
        fitted_optionValue = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, S_interior_tnsr:S_plot.reshape(-1,1)})
        
        # plot histogram of simulated process values and overlay estimated density
        #print(S_plot)
        #print(optionValue)
        plt.plot(S_plot, optionValue, color = 'b', label='Analytical Solution', linewidth = 3, linestyle=':')
        plt.plot(S_plot, fitted_optionValue[0], color = 'r', label='DGM estimate')    
        
        # subplot options
        plt.ylim(ymin=-1.0, ymax=3.0)
        plt.xlim(xmin=x_low, xmax=x_high)
        plt.xlabel(r"Space", fontsize=15, labelpad=10)
        plt.ylabel(r"Heat", fontsize=15, labelpad=20)
        plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(linestyle=':')
        
        if i == 0:
            plt.legend(loc='upper left', prop={'size': 16})
        
    # adjust space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    if saveFigure:
        plt.savefig(figureName)


if dim == 2:
    print("2 dimensions")

    name = "2d_heat_plot"

    valueTimes = np.linspace(t_low,t_high,9)
  
    # vector of t and S values for plotting
    S_plot = np.linspace(x_low, x_high, n_plot)
    S_plot = np.transpose([np.tile(S_plot,n_plot),np.repeat(S_plot,n_plot)])
    #print(np.shape(S_plot))
    #print(S_plot)

    upper_heat = 0

    for i, curr_t in enumerate(valueTimes):
        
        # specify subplot
        #plt.subplot(3,3,i+1)
        
        # simulate process at current t 
        # compute normalized density at all x values to plot and current t value
        t_plot = curr_t * np.ones((len(S_plot),1))
        fitted_optionValue = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, S_interior_tnsr:S_plot})
        fitted_optionValue = np.flip(np.reshape(fitted_optionValue,(n_plot,n_plot)),axis=0)

        #Testing config of S_plot
        #S_plot = np.reshape(S_plot,(n_plot,n_plot,2))
        #print(S_plot)
        if i == 0:
            upper_heat = np.amax(fitted_optionValue)

        # plot histogram of simulated process values and overlay estimated density
        #print(S_plot)
        #print(optionValue)
        plt.clf()
        print("plotting")
        plt.imshow(fitted_optionValue,cmap='hot',interpolation='nearest')
        plt.clim(0.0,upper_heat)
        plt.colorbar()
        plt.savefig("heat_multiDim/time " + str(curr_t) +  " " + figureName)
        # subplot options
        #plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
        
        #if i == 0:
           # plt.legend(loc='upper left', prop={'size': 16})
        

    #if saveFigure:
        #plt.savefig(figureName)