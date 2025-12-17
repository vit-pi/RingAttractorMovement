##
# IMPORTS
###
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import datetime
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap


###
# PROPERTIES
###


# Contains properties that define a behaviour (state of brain, speed, ...)
class BehProp:
    # initialize by providing parameters
    def __init__(self, kernel=(0, 10, 0.5), S=1, cue=None, T=1, tau=0.0, t_neuro=1.0, angle_num=64, v0=0.1,omega0=1,allo=True,init_pos=(0.,0.), init_head=0.0, init_hold=0.0, init_brain=(0,), N=1000, S_update_rule=[0]):
        """
        Initializes the parameters and initial configurations of neural and movement dynamics
        to model different scenarios of cognitive or behavioral activity.

        :param kernel: Defines the interaction between direction neurons as a tuple [type, parameters].
            Possible `type` values include:
            - 0: Cosine-shaped interaction with [J, nu], e.g., (0, 2, 0.5).
            - 1: Circular Mexican Hat function [J, kappa, nu], e.g., (1, 3, 1, 0.5).
            - 2: Shifted von Mises distribution with [J, kappa], e.g., (2, 1, 10).
        :type kernel: tuple

        :param S: Intrinsic stimulation.
        :type S: float

        :param cue: A list of environmental cues. Each cue element is defined as
            [type, random, kappa, strength, position], where:
            - `random` (bool): Indicates uniform random position generation (True/False).
            - `kappa` (float): Specifies the inverse width in von Mises distribution/width of a cue for spatially extended cues (type=3,4).
            - `strength` (float): Ranges between 0 to 1 to reflect overall cue strength.
            - `type` (int): Represents the type of cue:
                0 - Fixed cue; position defined by [pos_x, pos_y].
                1 - Another individual positioned at `ind_1`.
                2 - Fixed angle positioned at `theta_c`.
                3 - Fixed spatially extended cue; position defined by [pos_x, pos_y].
                    Note that instead of 'kappa' at index 2, this element encodes the radius of the cue.
                4 - a cue at [pos_x0, 0] is presented until the animal reaches x>=pos_x1, at which point a cue [pos_x1,pos_y] is presented for time `t_blink`, after which the cue disappears completely. So, position defined by: pos_x0, pos_x1, pos_y, t_blink, t_pass, t_set: where t_pass and t_set must be set to 0, t_pass start counting time after passing through pos_x0 and t_set remembers the crossing time.
        :type cue: Union(None,list)

        :param T: Neural noise.
        :type T: float

        :param tau: Neural transmission delay.
        :type tau: float

        :param t_neuro: Timescale of the neural dynamics.
        :type t_neuro: float

        :param angle_num: Number of angular positions (theta) in the configuration.
        :type angle_num: int

        :param N: Number of neural units per neural group.
        :type N: int

        :param v0: Speed of the animal.
        :type v0: float

        :param omega0: Characteristic angular velocity.
        :type omega0: float

        :param allo: Indicates the coordinate frame of reference:
            True for allocentric, False for egocentric.
        :type allo: bool

        :param init_pos: Initial position of the animal as (x, y).
        :type init_pos: tuple

        :param init_head: Initial heading angle measured in radians.
        :type init_head: float

        :param init_hold: Time duration during which the animal remains stationary,
            evolving only its neural dynamics.
        :type init_hold: float

        :param init_brain: Initializes the brain configuration with the following sub-types:
            - type=0 (brain_fp)
            - type=1 (fixed neural activity n0): [type, n0]
            - type=2 (fixed n0, and n1 within w of uniformly random cue): [type, n0, n1, w]
            - type=3 (fixed n0, and n1 within w of cue at angle theta): [type, n0, n1, w, theta]
            - type=4 (fixed n0, and n1 within w of cue at position x,y): [type, n0, n1, w, x, y]
            - type=5 (fixed n0, and n1 within w away from cue x,y): [type, n0, n1, w, x, y]
            - type=6 (brain fp, but initialize k groups of neural units with same value): [type, k]
        :type init_brain: tuple

        :param S_update_rule: Indicates how the intrinsic stimulation parameter S is updated during the simulation. Defined as [type, parameters]:
            - type=0: no updating
            - type=1: periodic update of S with frequency `omega` (float, param0)
            - type=2: feedback update of S: increase at rate `a` (flota, param0) per angular frequency of animal rotation, decay at rate `b` (float, param1)
        :type S_update_rule: list
        Attributes:
        - variables with names identical to the input parameters.
        + angle_step: Sampling distance / angular step along the circle, calculated as `2 * pi / angle_num` (float).
        + brain_fp: Stores the fixed point of brain activity (float).
        """
        # parameters of neural field model
        self.kernel = kernel                # interaction between direction neurons = [type, parameters]
                                            # [0,J,nu] for cosine-shaped interaction, [0, 2, 0.5]
                                            # [1,J,kappa] for circular mexican hat function, [1, 3, 1, 0.5]
                                            # [2,J,kappa] for shifted von Mises distribution [2, 1, 1O]
        self.S_init = S                     # intrinsic stimulation (initial value)
        self.S = S                          # intrinsic stimulation (possibly variable in time)
        if cue is None:                     # cue=[cue1,...], cue1=[type,random,kappa,mu,position]
            self.cue = []                   # random=t/f (true means position uniformly randomly generated when beh switched on)
        else:                               # kappa = 10 (parameter in the von Mises distribution - inverse of width)
            self.cue = cue                  # strength = between 0 and 1 (normalization of the von Mises distribution - reflects the strength)
                                            # type=0 (fixed cue), position=pos_x,pos_y
                                            # type=1 (another individual), position=ind_1
                                            # type=2 (fixed angle), position=theta_c
        self.T = T                          # neural noise
        self.tau = tau                      # neural transmission delay
        self.t_neuro = t_neuro              # timescale of the neural dynamics
        self.angle_num = angle_num          # number of theta positions
        self.angle_step = 2*np.pi/angle_num # number of points along a circle
        self.N = N                          # number of neural units per neural group
        # parameters in x,y equations
        self.v0 = v0                        # speed of the animal
        self.omega0 = omega0                # characteristic angular velocity
        self.allo = allo                    # coordinate frame of reference (allocentric or egocentric)
        # initial conditions
        self.init_pos = init_pos        # initial position of the animal
        self.init_head = init_head      # initial heading angle of the animal
        self.init_hold = init_hold      # time for which the animal does not move but only its neural dynamics evolves
        self.init_brain = init_brain    # [type,N=num neural units, n=default neural activity, others]
                                        # type=0 (brain_fp)
                                        # type=1 (fixed neural activity n0): [type, n0]
                                        # type=2 (fixed n0, and n1 within w of uniformly random cue): [type, n0, n1, w]
                                        # type=3 (fixed n0, and n1 within w of cue at angle theta): [type, n0, n1, w, theta]
                                        # type=4 (fixed n0, and n1 within w of cue at position x,y): [type, n0, n1, w, x, y]
                                        # type=5 (fixed n0, and n1 within w away from cue x,y): [type, n0, n1, w, x, y]
                                        # type=6 (brain fp, but initialize k groups of neural units with same value): [type, k]
        # update rule for the intrinsic stimulation parameter S
        self.S_update_rule = S_update_rule
        # save the fixed point of brain activity
        self.brain_fp = self.find_brain_fp()

    # interaction kernel J
    def find_J(self, theta):
        """
        Find the interaction kernel J(theta)

        :param theta: Angular difference.
        :type theta: float
        :return: interaction kernel (float)
        """
        J = 0
        if self.kernel[0] == 0:
            J = self.kernel[1] * np.cos(np.pi * (np.abs(theta) / np.pi) ** self.kernel[2]) * 2 * np.pi / self.angle_num
        elif self.kernel[0] == 1:
            J = self.kernel[1] * np.exp(self.kernel[2] * (np.cos(theta) - 1)) * (np.cos(theta) - self.kernel[2] * np.sin(theta) ** 2)
            J -= self.kernel[3]
            J *= 2*np.pi/self.angle_num
        elif self.kernel[0] == 2:
            norm = special.iv(0, self.kernel[2])
            J = np.exp(self.kernel[2] * np.cos(theta)) - norm
            J *= 2 * np.pi * self.kernel[1] / self.angle_num
        return J

    # Fourier modes of the interaction kernel J
    def find_Jk(self, k):
        """
        Find the discrete fourier modes of the interaction kernel J_k.

        :param k: wavenumber
        :type k: int

        :return:
        """
        J_k = 0
        for ang in range(self.angle_num):
            angle = ang*self.angle_step-np.pi
            J_k += self.find_J(angle)*np.cos(k*angle)
        return J_k

    # sigmoidal function
    def sigma(self, convo, stim):
        """
        Sigmoidal function used in the forcing of the neural dynamics.

        :param y: Value of the convolution (internal interactions)
        :type y: float

        :param stim: Value of the stimulation probability (external interactions).
        :type stim: float

        :return: sigma(y) (float).
        """
        return stim*self.S/(stim*self.S+np.exp((-self.find_J(0)/(2*self.N)-convo)/self.T))

    # find the homogeneous steady state of the brain model
    def find_brain_fp(self):
        """
        Locates the homogeneous steady state of the neural dynamics.

        :return: value of the homogeneous steady state (float)
        """
        J0 = self.find_Jk(0)
        if self.kernel[0] == 2:
            J0 = 0
        fun = lambda n: -n + self.sigma(n*J0,1)
        roots = optimize.root(fun,0).x[0]
        return roots

    # find effective interaction of the mode k
    def alpha(self,k):
        """
        Excitation number of the mode k.

        :param k: wavenumber
        :type k: int

        :return: excitation number (float).
        """
        return self.brain_fp*(1-self.brain_fp)*self.find_Jk(k)/self.T

    # update S
    def update_S(self, time, time_step, angular_velocity):
        """
        Update the value of the intrinsic stimulation S according to a prescribed rule.

        :param time: Current time.
        :type time: float
        :param time_step: Time step for numerical integration.
        :type time_step: float
        :param angular_velocity: Angular velocity of the animal.
        :type angular_velocity: float

        :return: None
        """
        if self.S_update_rule[0]==1:
            omega = self.S_update_rule[1]
            phi0 = np.arccos(2*self.S_init-1)
            self.S = (1+np.cos(omega*time+phi0))/2
        elif self.S_update_rule[0]==2:
            a = self.S_update_rule[1]
            b = self.S_update_rule[2]
            S0 = self.S_update_rule[3]
            self.S = max(integrate.solve_ivp(lambda t, S: a * angular_velocity - b * (S-S0), [0, time_step], [self.S]).y[0][-1], 0)
            # self.S += (a*angular_velocity-b*self.S)*time_step
            # print(self.S)

# Contains properties of an individual (behaviour and transitions between them)
class IndProp:
    # Initialize by providing beh_prop=[beh_prop1, ...]
    # and a matrix of transition rates [i][j] where the transition is from behaviour i to behaviour j
    def __init__(self, beh, trans=None, beh_stoch_switch=True):
        """
        Properties of an individual that can have multiple behaviors (BehProp) and switch between them either stochastically
        or deterministically.

        :param beh: list of behaviours (BehProp objects)
        :type beh: list[BehProp]

        :param trans: matrix of transition rates between behaviours, where the element at (i, j) represents the transition rate from behaviour `i` to `j`.
            By default, this matrix has zero elements and no transitions occur.
        :type trans: Union(None,numpy.ndarray[float, float])

        :param beh_stoch_switch: indicates whether the system should follow stochastic switching (default True) or deterministic switching between behaviors.
        :type beh_stoch_switch: bool


        Attributes:

        + variables with the same names as the input parameters (beh, beh_stoch_switch)

        + beh_num : int
            Total number of behaviors defined in the system.
        + beh_stoch_switch : bool
            Indicates whether switching between behaviors should follow a stochastic
            (True) or deterministic (False) approach.
        + rate: numpy.ndarray
            Total rate to move away from each behavior, calculated by summing
            transition rates from a given behavior to all others.
        + prob : numpy.ndarray
            Probability matrix indicating the likelihood of transitioning between
            behaviors. For an element at index (i, j), it represents the probability of moving
            from behavior `i` to `j`.
        """
        self.beh = beh                 # list of beh_prop
        self.beh_num = len(beh)        # total number of behaviours
        self.beh_stoch_switch = beh_stoch_switch        # True = stochastic switching between behaviours, False = deterministic switching
        if trans is None: # update the transition matrix if not provided
            trans = np.zeros((self.beh_num, self.beh_num))
        self.rate = np.sum(trans,axis=1)    # total rate to move away from behaviour i
        self.prob = np.zeros((self.beh_num,self.beh_num))   # probability to move from i to j given that moving from i
        for beh_i in range(self.beh_num):
            for beh_j in range(self.beh_num):
                if self.rate[beh_i] != 0:
                    self.prob[beh_i][beh_j] = trans[beh_i][beh_j]/self.rate[beh_i]

    def next_beh(self,beh_now):
        """
        Randomly sample the next behavior, given the current one.

        :param beh_now: index of the current behaviour on the self.beh list.
        :type beh_now: int

        :return: index of the next behaviour (int)
        """
        return np.random.choice(self.beh_num, p=self.prob[beh_now])

    def next_beh_time(self,beh_now):
        """
        Randomly sample the waiting time until switching to the next behaviour, given the current behaviour.

        :param beh_now: index of the current behaviour on the self.beh list.
        :type beh_now: int

        :return: waiting time until switching to the next behaviour (float)
        """
        if self.rate[beh_now] == 0:
            return np.inf
        else:
            if self.beh_stoch_switch:
                return -np.log(np.random.uniform(0,1))/self.rate[beh_now]
            else:
                return 1/self.rate[beh_now]

# Contains properties of an environment or shared by all individuals
class EnvProp:
    def __init__(self, ind_num=1,pos_max=(100,100),time_step=0.1,time_max=300,time_memory=300,angle_num=64):
        """
        Properties of the environment that are shared by all individuals.

        :param ind_num: Number of individuals in the simulation.
        :type ind_num: int

        :param pos_max: Maximum values of the coordinate system (max_x, max_y) used for sampling random cues of type '0'.
        :type pos_max: tuple

        :param time_max: Maximum simulation time.
        :type time_max: float

        :param time_step: Time step for numerical integration or simulation.
        :type time_step: float

        :param time_memory: Time span for states that are saved into the memory during the simulation.
        When time_memory is bigger than time_max, time_memory is set to time_max.
        :type time_memory: float

        :param angle_num: Number of angular discretizations in the system or simulation.
        :type angle_num: int

        Attributes:
        - variables with the same names as the input parameters (ind_num, pos_max, time_step, time_max, time_memory, angle_num).
        + angle_step: Angular step size, calculated as `2 * pi / angle_num`.
        """

        self.ind_num = ind_num              # number of individuals in a simulation
        self.pos_max = pos_max              # maximal x,y positions
        self.time_step = time_step          # time step in Euler method
        self.time_max = time_max            # maximal time of simulation
        self.time_memory = time_memory      # time duration kept in memory and used to plot a trajectory
        self.angle_num = angle_num          # number of theta positions
        self.angle_step = 2 * np.pi / self.angle_num  # number of points along a circle
        # ensure that the time_memory is smaller than the maximal time of the simulation
        if self.time_memory > self.time_max:
            self.time_memory = self.time_max

###
# PATTERN
###
# Main class that keeps the current state of simulation (pattern) and includes methods to update this pattern in time
# Input: EnvProp, IndProp (vector with IndProp objects for each individual)

class Pattern:

    ###
    # Properties and Pattern variables, Initialization
    ###
    # INPUT: env_prop, ind_prop=[ind_prop1,...]
    def __init__(self, env_prop, ind_prop):
        """
        Initializes the simulation object with the provided environment properties and individual properties.

        This constructor method handles deep copying of the properties, ensures consistency among individuals,
        and calculates the necessary time-related variables for the simulation. It also initializes pattern
        variables such as behavior states, positions, and brain-related configurations for individuals in the
        simulation. Additionally, this method verifies and adjusts the simulation's time step based on time
        delays and initializes behavior switching times.

        :param env_prop: Environment properties that define global simulation parameters, such as time step,
            memory length, and more.
        :type env_prop: EnvProp

        :param ind_prop: A list of individual properties which contain details about the initial states, behavioral
            attributes, and internal configurations of each individual in the simulation.
        :type ind_prop: list[IndProp]
        """

        # properties:
        self.ind_prop = copy.deepcopy(ind_prop)
        self.env_prop = copy.deepcopy(env_prop)
        self.env_prop.ind_num = len(self.ind_prop)
        # make sure that all individuals have the same angle_num and angle_step, given by env_prop
        for ind in range(self.env_prop.ind_num):
            for beh in self.ind_prop[ind].beh:
                beh.angle_num = self.env_prop.angle_num
                beh.angle_step = self.env_prop.angle_step
        # length of saved times, readjust time_step to make it smaller than the smallest nonzero time delay
        # when doing stochastic simulations, also manually make sure that tau<t_neuro/(2*N*angle_num)
        taus = np.asarray([np.asarray([beh.tau for beh in ind.beh]) for ind in self.ind_prop])
        if np.max(taus) > 0:
            tau_min = np.min(taus[np.nonzero(taus)])
            self.env_prop.time_step = min(tau_min/4,self.env_prop.time_step)
        self.time_num = int(np.ceil(max(np.max(taus), self.env_prop.time_memory) / self.env_prop.time_step)) + 1
        # create pointers to times
        self.pt_now = self.time_num - 1
        self.pt_tau = (self.time_num*np.ones(self.env_prop.ind_num)-np.ceil(taus[:,0]/self.env_prop.time_step)-1).astype(int)
        # create pattern variables
        self.behaviour = np.zeros(self.env_prop.ind_num, dtype=int)        # initially, each individual has the "0" behaviour
        self.nx = np.zeros((self.time_num, self.env_prop.ind_num, self.env_prop.angle_num + 3))
        self.S_vals = np.zeros((self.env_prop.ind_num,self.time_num))
        self.time = 0
        # create variable for next behaviour switch
        self.next_beh_time = np.zeros(self.env_prop.ind_num)
        for ind in range(self.env_prop.ind_num):
            self.next_beh_time[ind] = self.ind_prop[ind].next_beh_time(0)    # generate next behaviour time, having started at behaviour 0
        # save the variables of the behavioural Markov process
        self.beh_seq = [[[0,0]] for ind in range(self.env_prop.ind_num)]
        # initialize pattern variables
        self.nx[:] = np.nan
        for tim in range(self.time_num):
            for ind in range(self.env_prop.ind_num):
                for ang in range(self.env_prop.angle_num):
                    # initialized at brain_fp
                    if self.find_ind(ind).init_brain[0] == 0:
                        self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).brain_fp)/self.find_ind(ind).N
                    # intialized at fixed n0
                    if self.find_ind(ind).init_brain[0] == 1:
                        self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[1])/self.find_ind(ind).N
                    # initialized at fixed n0, and n1 within w of random cue
                    if self.find_ind(ind).init_brain[0] == 2:
                        if tim==0 and ind==0 and ang==0:
                            self.find_ind(ind).init_brain.append(np.random.uniform(0,2*np.pi))
                        dtheta = np.abs(ang*self.env_prop.angle_step - self.find_ind(ind).init_brain[4]) % (2 * np.pi)  # relative angle between chosen angle and direction away from cue
                        dtheta = min(dtheta, 2 * np.pi - dtheta)  # relative angle in range [0,np.pi]
                        if dtheta < self.find_ind(ind).init_brain[3] / 2:  # if angle within the width of cue
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[2])/self.find_ind(ind).N
                        else:
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[1])/self.find_ind(ind).N
                    # initialized at fixed n0, and n1 within w of cue at angle theta
                    if self.find_ind(ind).init_brain[0] == 3:
                        dtheta = np.abs(ang*self.env_prop.angle_step - self.find_ind(ind).init_brain[4]) % (2 * np.pi)  # relative angle between chosen angle and direction away from cue
                        dtheta = min(dtheta, 2 * np.pi - dtheta)  # relative angle in range [0,np.pi]
                        if dtheta < self.find_ind(ind).init_brain[3] / 2:  # if angle within the width of cue
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[2])/self.find_ind(ind).N
                        else:
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[1])/self.find_ind(ind).N
                    # initialized at fixed n0, and n1 within w of cue at position x,y
                    if self.find_ind(ind).init_brain[0] == 4:
                        dx = self.find_ind(ind).init_brain[4] - self.find_ind(ind).init_pos[0]
                        dy = self.find_ind(ind).init_brain[5] - self.find_ind(ind).init_pos[1]
                        theta_0 = np.arctan2(dy, dx) % (2 * np.pi)
                        dtheta = np.abs(ang*self.env_prop.angle_step - theta_0) % (2 * np.pi)  # relative angle between chosen angle and direction away from cue
                        dtheta = min(dtheta, 2 * np.pi - dtheta)  # relative angle in range [0,np.pi]
                        if dtheta < self.find_ind(ind).init_brain[3] / 2:  # if angle within the width of cue
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[2])/self.find_ind(ind).N
                        else:
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[1])/self.find_ind(ind).N
                    # initialized at fixed n0, and n1 within w away from a cue at position x,y
                    if self.find_ind(ind).init_brain[0] == 5:
                        dx = self.find_ind(ind).init_brain[4] - self.find_ind(ind).init_pos[0]
                        dy = self.find_ind(ind).init_brain[5] - self.find_ind(ind).init_pos[1]
                        theta_0 = (np.arctan2(dy, dx)+np.pi) % (2 * np.pi)
                        dtheta = np.abs(ang*self.env_prop.angle_step - theta_0) % (2 * np.pi)  # relative angle between chosen angle and direction away from cue
                        dtheta = min(dtheta, 2 * np.pi - dtheta)  # relative angle in range [0,np.pi]
                        if dtheta < self.find_ind(ind).init_brain[3] / 2:  # if angle within the width of cue
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[2])/self.find_ind(ind).N
                        else:
                            self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(self.find_ind(ind).N,self.find_ind(ind).init_brain[1])/self.find_ind(ind).N
                    # initialized at brain_fp but k groups with the same value
                    if self.find_ind(ind).init_brain[0] == 6:
                        trials = np.floor(self.find_ind(ind).N/self.find_ind(ind).init_brain[1])
                        self.nx[self.time_num - tim - 1][ind][ang] = np.random.binomial(trials,self.find_ind(ind).brain_fp)/trials
                self.nx[self.time_num-tim-1][ind][self.env_prop.angle_num] = self.find_ind(ind).init_pos[0]
                self.nx[self.time_num-tim-1][ind][self.env_prop.angle_num+1] = self.find_ind(ind).init_pos[1]
                self.nx[self.time_num-tim-1][ind][self.env_prop.angle_num+2] = self.find_ind(ind).init_head
                self.S_vals[ind][self.time_num-tim-1] = self.find_ind(ind).S_init

    ###
    # Measurements
    ###
    # find self.ind_prop[ind].beh[self.behaviour[ind]] to directly access the properties of the current behaviour
    def find_ind(self,ind):
        """
        Current behaviour (BehProp) of a given individual with a specified index.

        :param ind: Individual index.
        :type ind: int

        :return: Behaviour properties (BehProp).
        """

        return self.ind_prop[ind].beh[self.behaviour[ind]]

    # compute integral of J(theta-theta')n(theta',t-tau) from 0 to 2*pi
    def find_convolution(self, ind, ang):
        """
        Find the convolution of the neural activity and the interaction kernel.

        :param ind: Index of the individual.
        :type ind: int

        :param ang: Index of an angle at which we perform the convolution.
        :type ang: int

        :return: convolution (float).
        """
        conv = 0
        for ang_i in range(self.env_prop.angle_num):
            ang_diff = abs(ang-ang_i)
            if ang_diff <= self.env_prop.angle_num // 2:
                ang_diff = 2*np.pi*ang_diff/self.env_prop.angle_num
            else:
                ang_diff = 2 * np.pi * (self.env_prop.angle_num - ang_diff) / self.env_prop.angle_num
            conv += self.find_ind(ind).find_J(ang_diff)*self.nx[self.pt_tau[ind]][ind][ang_i]
        return conv

    # find the angle of a given cue
    def find_theta_c_kappa(self, ind, cue):
        """
        Find the angle of a given cue.

        :param ind: Index of the individual.
        :type ind: int

        :param cue: A specified cue.
        :type cue: list

        :return: angle of the cue (float), kappa of the cue (float).
        """
        theta_c = 0
        kappa = cue[2]
        if cue[0] == 0:  # fixed stimulus
            x = cue[4]
            y = cue[5]
            dx = x - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num]  # x distance from stimulus
            dy = y - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num + 1]  # y distance from stimulus
            theta_c = np.arctan2(dy, dx)  # angle from stimulus
        elif cue[0] == 1:  # stimulus of a moving individual
            target_ind = cue[4]
            x = self.nx[self.pt_tau[target_ind]][target_ind][self.env_prop.angle_num]
            y = self.nx[self.pt_tau[target_ind]][target_ind][self.env_prop.angle_num + 1]
            dx = x - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num]  # x distance from stimulus
            dy = y - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num + 1]  # y distance from stimulus
            theta_c = np.arctan2(dy, dx)  # angle from stimulus
        elif cue[0] == 2:  # stimulus at a fixed angle
            theta_c = cue[4]
        elif cue[0] == 3:  # fixed spatially extended stimulus
            x = cue[4]
            y = cue[5]
            dx = x - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num]  # x distance from stimulus
            dy = y - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num + 1]  # y distance from stimulus
            theta_c = np.arctan2(dy, dx)  # angle from stimulus
            d = np.sqrt(dx**2+dy**2)
            w = cue[2]
            if d < w:
                d = w
            kappa = (np.pi/np.arcsin(w/d)-1/np.pi)/2
        elif cue[0] == 4:
            if self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num]<cue[5]:
                x = cue[4]
                y = 0
                dx = x - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num]  # x distance from stimulus
                dy = y - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num + 1]  # y distance from stimulus
                theta_c = np.arctan2(dy, dx)  # angle from stimulus
                d = np.sqrt(dx**2+dy**2)
                w = cue[2]
                if d < w:
                    d = w
                kappa = (np.pi/np.arcsin(w/d)-1/np.pi)/2
            else:
                x = cue[5]
                y = cue[6]
                dx = x - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num]  # x distance from stimulus
                dy = y - self.nx[self.pt_tau[ind]][ind][self.env_prop.angle_num + 1]  # y distance from stimulus
                theta_c = np.arctan2(dy, dx)  # angle from stimulus
                d = np.sqrt(dx ** 2 + dy ** 2)
                w = cue[2]
                if d < w:
                    d = w
                kappa = (np.pi / np.arcsin(w / d) - 1 / np.pi) / 2

        return theta_c, kappa

    # compute contribution to v equation due to stimulation I
    def find_stimulation(self, ind, ang):
        """
        Find the stimulation probability s(theta,t) of a given individual at a given angle.

        :param ind: Index of the individual.
        :type ind: int
        :param ang: Index of the angle.
        :type ang: int

        :return: stimulation probability (float).
        """
        theta = 2 * np.pi * ang / self.env_prop.angle_num   # get chosen angle in radians
        stim = 1        # default stimulation probability
        inh_cue = 1    # stimulation coming from the cues
        for cue in self.find_ind(ind).cue:  # add each cue
            theta_c,kappa = self.find_theta_c_kappa(ind, cue)  # angle of the cue
            if cue[0] == 4:
                if cue[8]<cue[7]:
                    inh_cue *= 1 - cue[3] * np.exp(kappa * (np.cos(theta - theta_c) - 1))
                else:
                    inh_cue = 0
            else:
                inh_cue *= 1-cue[3]*np.exp(kappa*(np.cos(theta-theta_c)-1))
        if len(self.find_ind(ind).cue)>0:
            stim -= inh_cue
        return stim

    # compute forcing for a given n[spec][diff][pos] variable, already multiplied by time-step
    def find_forcing_brain(self, ind, ang):
        """
        Find the forcing of the brain activity of a given individual at a given angle, already multiplied by time-step

        :param ind: Index of the individual.
        :type ind: int
        :param ang: Index of the angle.
        :type ang: int

        :return: forcing (float).
        """
        force = -self.nx[self.pt_now][ind][ang]
        force += self.find_ind(ind).sigma(self.find_convolution(ind,ang),self.find_stimulation(ind, ang))
        force *= self.env_prop.time_step/self.find_ind(ind).t_neuro
        return force

    # find velocity of individual ind from neural activity
    def find_velocity(self, ind):
        """
        Find the velocity of an individual.

        :param ind: Index of the individual.
        :type ind: int

        :return: velocity (float).
        """
        head_angle = self.nx[self.pt_now][ind][self.env_prop.angle_num+2]
        vel_x = self.find_ind(ind).v0*np.cos(head_angle)
        vel_y = self.find_ind(ind).v0*np.sin(head_angle)
        return vel_x, vel_y

    # find velocity of individual ind from neural activity
    def find_angular_velocity(self, ind):
        """
        Find the angular velocity of an individual.

        :param ind: Index of the individual.
        :type ind: int

        :return: angular velocity (float).
        """
        # calculate the vector of the neural activity
        n_x = 0
        n_y = 0
        # compute integral of cos(theta)n(theta,t) (resp. sin(theta)n(theta,t)) from 0 to 2*pi
        for ang in range(self.env_prop.angle_num):
            n_x += self.nx[self.pt_now][ind][ang]*np.cos(ang*2*np.pi/self.env_prop.angle_num)
            n_y += self.nx[self.pt_now][ind][ang]*np.sin(ang*2*np.pi/self.env_prop.angle_num)
        # rotate the neural activity to the allocentric coordinates if the animal works in egocentric coordinates
        n_x *= 2*np.pi/self.env_prop.angle_num
        n_y *= 2*np.pi/self.env_prop.angle_num
        head_angle = self.nx[self.pt_now][ind][self.env_prop.angle_num + 2]
        if not self.find_ind(ind).allo:
            n_x, n_y = np.cos(head_angle)*n_x - np.sin(head_angle)*n_y, np.sin(head_angle)*n_x + np.cos(head_angle)*n_y
        # find the angular velocity
        ang_vel = self.find_ind(ind).omega0*(n_y*np.cos(head_angle)-n_x*np.sin(head_angle))
        return ang_vel

    # find forcing in x
    def find_forcing_move(self, ind):
        """
        Find the forcing for the movement of a given individual.

        :param ind: Index of the individual.
        :type ind: int

        :return: A tuple that contains the x and y components of the movement force (float, float), as well as the force that rotates the heading direction (float).
        """
        if self.time < self.find_ind(ind).init_hold:
            return 0, 0, 0
        else:
            vel_x, vel_y = self.find_velocity(ind)
            ang_vel = self.find_angular_velocity(ind)
            return vel_x * self.env_prop.time_step, vel_y * self.env_prop.time_step, ang_vel * self.env_prop.time_step

    # find stochastic transition rates
    def find_rate(self, ind, ang):
        """
        Find the stochastic transition rates for a given individual and an angle.

        :param ind: Index of the individual.
        :type ind: int

        :param ang: Index of the angle.
        :type ang: int

        :return: rate_up (float), rate_down (float)
        """
        rate_up = self.find_ind(ind).sigma(self.find_convolution(ind,ang),self.find_stimulation(ind, ang))
        rate_down = 1-rate_up
        rate_up *= self.find_ind(ind).N*(1-self.nx[self.pt_now][ind][ang])/self.find_ind(ind).t_neuro
        rate_down *= self.find_ind(ind).N*self.nx[self.pt_now][ind][ang]/self.find_ind(ind).t_neuro
        return rate_up, rate_down

    # save all properties in a text file at a given location
    def save_properties(self, location, *name):
        """
        Saves the properties of the environment and the individuals in a text file.

        :param location: Location of the text file.
        :type location: str
        :param name: Name of the text file.
        :type name: str
        """
        e_prop = [dict([(x, y) for x, y in vars(self.env_prop).items() if not x.startswith("_")])]
        i_prop = []
        for ind in self.ind_prop:
            i_prop.append(dict([(x, y) for x, y in vars(ind).items() if not x.startswith("_")]))
            for beh in ind.beh:
                i_prop.append(dict([(x, y) for x, y in vars(beh).items() if not x.startswith("_")]))
        if len(name)>0:
            pd.DataFrame(e_prop).to_csv(location+'//'+name[0]+"env_prop.csv", index=False)
            pd.DataFrame(i_prop).to_csv(location + '//'+name[0]+"ind_prop.csv", index=False)
        else:
            pd.DataFrame(e_prop).to_csv(location + '//env_prop.csv', index=False)
            pd.DataFrame(i_prop).to_csv(location + '//ind_prop.csv', index=False)

    ###
    # Time-evolution of pattern variables
    ###

    # update randomly generated stimuli
    def update_random_cues(self, ind):
        """
        Update the positions of randomly generated cues.

        :param ind: Index of the individual.
        :type ind: int
        """
        for cue in self.find_ind(ind).cue: # look at each cue
            if cue[1]:          # see if it is randomly updated
                if cue[0]==0:     # fixed allocentric position
                    cue[5] = np.random.uniform(0,self.env_prop.pos_max[0])
                    cue[6] = np.random.uniform(0,self.env_prop.pos_max[1])
                elif cue[0]==1:   # fixed individual
                    cue[5] = np.random.choice(self.env_prop.ind_num)
                elif cue[0]==2:   # fixed angle
                    cue[5] = np.random.uniform(0,2*np.pi)

    # update behaviour of each inidividual
    def update_behaviour(self):
        """
        Update the behaviour of each individual.
        """
        for ind in range(self.env_prop.ind_num):
            # update the behaviour if the time that will be now updated in update_pattern is after or equal to the waiting time
            if self.time+self.env_prop.time_step >= self.next_beh_time[ind]:
                # update behaviour
                old_behaviour = self.behaviour[ind]
                self.behaviour[ind] = self.ind_prop[ind].next_beh(self.behaviour[ind])
                # update beh_seq
                self.beh_seq[ind].append([self.next_beh_time[ind], self.behaviour[ind]])
                # update next time to switch behaviour
                self.next_beh_time[ind] += self.ind_prop[ind].next_beh_time(self.behaviour[ind])
                # update tau pointer for the new behaviour
                self.pt_tau[ind] = int(self.pt_now-np.ceil(self.find_ind(ind).tau/self.env_prop.time_step))
                # update randomly generated stimuli associated with the new behaviour
                self.update_random_cues(ind)
                # rotate the neural activity if the behaviour switch involves switching between allocentric and egocentric coordinates
                old_allo = self.ind_prop[ind].beh[old_behaviour].allo
                new_allo = self.ind_prop[ind].beh[self.behaviour[ind]].allo
                if old_allo and not new_allo:
                    # change the coordinates of a given individual to egocentric coordinates for all times
                    for tim in range(self.time_num):
                        phi = self.nx[tim][ind][self.env_prop.angle_num+2] % (2 * np.pi)
                        roll_angle = -round(phi/self.env_prop.angle_step)
                        self.nx[tim,ind,:self.env_prop.angle_num]=np.roll(self.nx[tim,ind,:self.env_prop.angle_num],roll_angle,axis=0)
                elif not old_allo and new_allo:
                    # change the coordinates of a given individual to allocentric coordinates for all times
                    for tim in range(self.time_num):
                        phi = self.nx[tim][ind][self.env_prop.angle_num+2] % (2 * np.pi)
                        roll_angle = round(phi/self.env_prop.angle_step)
                        self.nx[tim,ind,:self.env_prop.angle_num]=np.roll(self.nx[tim,ind,:self.env_prop.angle_num],roll_angle,axis=0)

    # update pattern with Euler method (deterministc=True), or the stochastic algorithm (deterministc=False)
    def update_pattern(self, deterministic=True):
        """
        Update the state variables.

        :param deterministic: use the Euler method (deterministic=True, default) or the stochastic algorithm (deterministic=False)
        :type deterministic: bool
        """
        # update behaviour
        self.update_behaviour()
        # update pattern variables
        pt_new = (self.pt_now + 1) % self.time_num
        nx_new = self.nx.copy()
        for ind in range(self.env_prop.ind_num):
            # update neural field
            if deterministic:
                for ang in range(self.env_prop.angle_num):
                    nx_new[pt_new][ind][ang] = self.nx[self.pt_now][ind][ang]+self.find_forcing_brain(ind,ang)
            else:
                # update all angles
                nx_new[pt_new][ind][:self.env_prop.angle_num] = self.nx[self.pt_now][ind][:self.env_prop.angle_num].copy()
                # update chosen angle
                ang = np.random.randint(0, self.env_prop.angle_num) # choose random angle
                up_rate, down_rate = self.find_rate(ind,ang) # compute transition rates
                rand = np.random.uniform() # flip a coin to decide on the transition
                if rand<self.env_prop.angle_num*self.env_prop.time_step*up_rate:
                    nx_new[pt_new][ind][ang] = self.nx[self.pt_now][ind][ang]+1/self.find_ind(ind).N
                elif rand<self.env_prop.angle_num*self.env_prop.time_step*(up_rate+down_rate):
                    nx_new[pt_new][ind][ang] = self.nx[self.pt_now][ind][ang]-1/self.find_ind(ind).N
            # update position
            force_x, force_y, force_phi = self.find_forcing_move(ind)
            nx_new[pt_new][ind][self.env_prop.angle_num] = self.nx[self.pt_now][ind][self.env_prop.angle_num] + force_x
            nx_new[pt_new][ind][self.env_prop.angle_num+1] = self.nx[self.pt_now][ind][self.env_prop.angle_num+1] + force_y
            nx_new[pt_new][ind][self.env_prop.angle_num+2] = self.nx[self.pt_now][ind][self.env_prop.angle_num+2] + force_phi
            # update stimulation strength of the individual - use the fact that force_phi is the angular velocity*time_step
            self.find_ind(ind).update_S(self.time,self.env_prop.time_step,force_phi/self.env_prop.time_step)
            self.S_vals[ind][pt_new] = self.find_ind(ind).S
            # update cue if time dependent
            for cue in self.find_ind(ind).cue:
                if cue[0] == 4:
                    if self.nx[self.pt_now][ind][self.env_prop.angle_num]>=cue[5]:
                        if cue[8] == 0:
                            cue[9] = self.time
                        cue[8] += self.env_prop.time_step # update time of the "ticking bomb" - i.e. when cue disappears
        self.nx = nx_new
        # update pointers
        self.pt_now = (self.pt_now + 1) % self.time_num
        self.pt_tau = (self.pt_tau + 1) % self.time_num
        # update self.time
        self.time += self.env_prop.time_step

    # run the simulation
    def simulate(self, deterministic=True):
        """
        Run the simulation.

        :param deterministic: use the Euler method (deterministic=True, default) or the stochastic algorithm (deterministic=False)
        :type deterministic: bool
        """
        percentage = 0
        while self.time <= self.env_prop.time_max:
            if self.time >= percentage * self.env_prop.time_max / 100:
                print("Simulation finished up to" + str(percentage) + "%")
                percentage += 1
            self.update_pattern(deterministic)

    ###
    # Get data for plotting
    ###

    # behaviour switch
    def get_behaviour_switch(self,ind,S_crit=0.16):
        """
        Find the times and behaviours at which the behaviour switches.
        Focus on times between (floor(t_max/t_step)-t_num+1)*t_step and floor(t_max/t_step)*t_step and include the endpoints even if there is no switch.
        Run 'simulate' method first before applying this function.
        Note that for S_update_rule[0]=0 (i.e. S unchanged), the behaviour switches are identified.
        However, if S_update_rule[0]!=0 (i.e. S changes), it is assumed that the individual consists of a single
        behaviour object and instead the behaviours are identified based on the value of S being above S_crit (0) or below S_crit (1).

        :return: switch_times (list), switch_t_index (list), switch_behaviour (list)
        """
        # get the minimal and maximal time
        t_max = np.floor(self.env_prop.time_max / self.env_prop.time_step) * self.env_prop.time_step
        t_min = t_max - (self.time_num - 1) * self.env_prop.time_step
        # trim the beh_seq to include only times between these bounds (inclusive)
        beh_seq = []
        init_beh = 0
        # blinking cue (transition from decision-making to search)
        if len(self.find_ind(ind).cue) > 0 and self.find_ind(ind).cue[0][0] == 4:
            beh_seq.append([t_min, 0])
            beh_seq.append([self.find_ind(ind).cue[0][9] + self.find_ind(ind).cue[0][7], 1])
            beh_seq.append([t_max + self.env_prop.time_step, 1])
        # stochastic transitions
        elif self.find_ind(ind).S_update_rule[0]==0:
            for beh in self.beh_seq[ind]:
                if beh[0]>=t_min:
                    if beh[0]<t_max:
                        beh_seq.append(beh)
                else:
                    init_beh = beh[1]
            # add the endpoints
            if len(beh_seq)==0:
                beh_seq.append([t_min, init_beh])
            elif beh_seq[0][0]!=t_min:
                beh_seq.insert(0,[t_min, init_beh])
            # add the endpoints
            beh_seq[0] = [t_min,beh_seq[0][1]]
            beh_seq.append([t_max+self.env_prop.time_step,beh_seq[-1][1]])
        # transitions due to variable S
        else:
            # get s values and times
            times, s_vals = self.get_S_trajectory(ind)
            mask = (times >= t_min) & (times <= t_max)
            times = times[mask]
            s_vals = s_vals[mask]
            # get the first and final behaviour
            final_beh = 0
            if s_vals[0]<S_crit:
                init_beh = 1
            if s_vals[-1]>S_crit:
                final_beh = 1
            # create the beh_seq
            beh_seq.append([t_min, init_beh])
            for i in range(len(times)):
                if s_vals[i]<S_crit and beh_seq[-1][1]==0:
                    beh_seq.append([times[i], 1])
                if s_vals[i]>S_crit and beh_seq[-1][1]==1:
                    beh_seq.append([times[i], 0])
            beh_seq.append([t_max+self.env_prop.time_step, final_beh])
        # get the output variables
        switch_times = [beh[0] for beh in beh_seq] # times at which the behaviour switches
        switch_t_index = [int(np.ceil((beh[0]-t_min)/self.env_prop.time_step)) for beh in beh_seq] # first time index that has been updated with the new behaviour
        switch_behaviour = [beh[1] for beh in beh_seq] # behaviour indices into which behaviour switches at the prescribed times
        return switch_times, switch_t_index, switch_behaviour

    # heading trajectory
    def get_heading_trajectory(self,ind):
        """
        Find the heading trajectory, i.e., heading angles over time.
        Run 'simulate' method first before applying this function.

        :param ind: Index of the individual.
        :type ind: int

        :return: times (np.ndarray), headings (np.ndarray)
        """
        # get times
        t_max = np.floor(self.env_prop.time_max / self.env_prop.time_step) * self.env_prop.time_step
        t_min = t_max - (self.time_num - 1) * self.env_prop.time_step
        times = np.linspace(t_min, t_max, self.time_num)
        # get headings
        headings = np.roll(self.nx[:, ind,self.env_prop.angle_num+2], -self.pt_now - 1, axis=0)
        return times, headings

    # movement trajectory
    def get_movement_trajectory(self,ind):
        """
        Find the movement trajectory, i.e., x and y coordinates over time.
        Run 'simulate' method first before applying this function.

        :param ind: Index of the individual.
        :type ind: int

        :return: times (np.ndarray), x (np.ndarray), y (np.ndarray)
        """
        # get times
        t_max = np.floor(self.env_prop.time_max / self.env_prop.time_step) * self.env_prop.time_step
        t_min = t_max - (self.time_num - 1) * self.env_prop.time_step
        times = np.linspace(t_min, t_max, self.time_num)
        # get headings
        x = np.roll(self.nx[:, ind,self.env_prop.angle_num], -self.pt_now - 1, axis=0)
        y = np.roll(self.nx[:, ind, self.env_prop.angle_num+1], -self.pt_now - 1, axis=0)
        return times, x, y

    # neural trajectory
    def get_brain_trajectory(self, ind, allocentric=True):
        """
        Get the trajectory of the neural dynamics for a given individual in prescribed coordinate system.
        Focus on times between (floor(t_max/t_step)-t_num+1)*t_step and floor(t_max/t_step)*t_step.
        Run 'simulate' method first before applying this function.

        :param ind: Index of individual.
        :type ind: int

        :param allocentric: use allocentric coordinates (default=True) or egocentric coordinates (allocentric=False)
        :type allocentric: bool

        :returns: A tuple that contains nd.arrays: times (x axis), thetas (y axis), and the neural activity (heatmap on these axes).
        """
        # get neural dynamics trajectory by rolling the latest point to the final entry
        traj = np.roll(self.nx[:, ind, :self.env_prop.angle_num], -self.pt_now - 1, axis=0)
        # get axis for angles
        thetas = np.array([k*self.env_prop.angle_step for k in range(self.env_prop.angle_num)])
        # get axis for times and find headings
        times, headings = self.get_heading_trajectory(ind)
        # determine whether the final behaviour is allocentric or egocentric
        final_allo = self.find_ind(ind).allo
        # change to the appropriate coordinates
        if final_allo and not allocentric: # want to output egocentric coordinates but the data is in allocentric coordinates
            # change the coordinates of a given individual to egocentric coordinates for all times
            for tim in range(self.time_num):
                phi = headings[tim] % (2 * np.pi)
                roll_angle = -round(phi/self.env_prop.angle_step)
                traj[tim]=np.roll(traj[tim],roll_angle)
        elif not final_allo and allocentric: # want to output allocentric coordinates but the data is in egocentric coordinates
            # change the coordinates of a given individual to allocentric coordinates for all times
            for tim in range(self.time_num):
                phi = headings[tim] % (2 * np.pi)
                roll_angle = round(phi/self.env_prop.angle_step)
                traj[tim]=np.roll(traj[tim],roll_angle)
        # return the trajectory
        return times, thetas, traj

    # S trajectory
    def get_S_trajectory(self, ind):
        """
        Find the trajectory of the intrinsic stimulation S over time.
        Run 'simulate' method first before applying this function.

        :param ind: Index of the individual.
        :type ind: int

        :return: times (np.ndarray), s_vals (np.ndarray)
        """
        # get times
        t_max = np.floor(self.env_prop.time_max / self.env_prop.time_step) * self.env_prop.time_step
        t_min = t_max - (self.time_num - 1) * self.env_prop.time_step
        times = np.linspace(t_min, t_max, self.time_num)
        # get headings
        s_vals = np.roll(self.S_vals[ind], -self.pt_now - 1)
        return times, s_vals

    # get distance to a nearest cue of type 0
    def get_cue_distance(self, ind):
        """
        Get the distance to the nearest cue of type 0 or 3.
        Run 'simulate' method first before applying this function.

        :param ind: Index of the individual.
        :type ind: int
        :return: distance (float)
        """
        x = self.nx[self.pt_now, ind, self.env_prop.angle_num]
        y = self.nx[self.pt_now,ind,self.env_prop.angle_num+1]
        distance = np.inf
        for beh in self.ind_prop[ind].beh:
            for cue in beh.cue:
                if cue[0]==0 or cue[0]==3:
                    distance = np.min([distance, np.sqrt((cue[4]-x)**2+(cue[5]-y)**2)])
        return distance

    # get minimal distance across all individuals
    def get_minimal_cue_distance(self):
        """
        Get the minimal distance to a nearest cue of type 0 across all individuals.
        Run 'simulate' method first before applying this function.
        :return: distance (float)
        """
        distance = np.inf
        for ind in range(self.env_prop.ind_num):
            distance = np.min([distance, self.get_cue_distance(ind)])
        return distance

    ###
    # Calculate statistics of the trajectories
    ###
    # get autocorrelation for a given time lag
    def find_autocorrelation(self, ind, t_lag):
        """
        Calculate autocorrelation with a specified t_lag for a given individual and its trajectory.

        :param ind: Index of the individual.
        :type ind: int
        :param t_lag: Length of time lag.
        :type t_lag: float
        :return: mean (float), standard deviation (float) of the autocorrelation.
        """
        # Get trajectory
        time, x, y = self.get_movement_trajectory(ind)
        # Calculate velocity components at each time step
        vx = np.diff(x) / np.diff(time)
        vy = np.diff(y) / np.diff(time)
        velocities = np.vstack((vx, vy)).T  # (N-1, 2)
        # Convert t_lag into the number of time steps
        lag_steps = int(t_lag / self.env_prop.time_step)
        # If t_lag exceeds the available trajectory length
        if lag_steps < 0 or lag_steps >= velocities.shape[0]:
            raise ValueError("t_lag is out of bounds for the trajectory.")
        # Calculate autocorrelations for all valid time points
        autocorrelations = np.zeros(velocities.shape[0] - lag_steps)
        for t in range(velocities.shape[0] - lag_steps):
            v_t = velocities[t]
            v_t_lag = velocities[t + lag_steps]
            autocorrelations[t] = np.dot(v_t, v_t_lag) / self.find_ind(ind).v0**2
        # Calculate mean and standard deviation
        mean_autocorr = np.mean(autocorrelations)
        std_autocorr = np.std(autocorrelations)
        return mean_autocorr, std_autocorr

    # get autocorrelation for all time lags up to t_lag_max
    def get_autocorrelations_all(self, ind, t_lag_max):
        """
        Calculate autocorrelation for all time lags up to t_lag_max for a given individual and its trajectory.

        :param ind: Index of the individual.
        :type ind: int
        :param t_lag_max: Maximum length of time lag.
        :type t_lag_max: float
        :return: mean (float), standard deviation (float) of the autocorrelation.
        """
        # Convert t_lag into the number of time steps
        lag_steps_max = int(np.floor(t_lag_max / self.env_prop.time_step)+1)
        # Maximal number of sampled points
        num = min(100, lag_steps_max)
        # Initialize autocorrelation array
        mean = np.zeros(num+1)
        std = np.zeros(num+1)
        time_lags = np.zeros(num+1)
        # Calculate autocorrelations for all valid time points
        for n in range(num+1):
            lag_steps = int(n * lag_steps_max / num)
            mean_autocorr, std_autocorr = self.find_autocorrelation(ind, lag_steps * self.env_prop.time_step)
            mean[n] = mean_autocorr
            std[n] = std_autocorr
            time_lags[n] = lag_steps * self.env_prop.time_step
        return time_lags, mean, std

    # get mean squared displacement for a given time lag
    def find_mean_squared_displacement(self, ind, t_lag):
        """
        Calculate mean-squared displacement for a specified t_lag for a given individual and its trajectory.

        :param ind: Index of the individual.
        :type ind: int
        :param t_lag: Length of time lag.
        :type t_lag: float
        :return: mean (float), standard deviation (float) of the autocorrelation.
        """
        # Get trajectory
        time, x, y = self.get_movement_trajectory(ind)
        # Convert t_lag into the number of time steps
        lag_steps = int(t_lag / self.env_prop.time_step)
        # If t_lag exceeds the available trajectory length
        if lag_steps < 0 or lag_steps >= time.size:
            raise ValueError("t_lag is out of bounds for the trajectory.")
        # Calculate autocorrelations for all valid time points
        msd = np.zeros(time.size - lag_steps)
        for t in range(time.size - lag_steps):
            msd[t] = ((x[t+lag_steps] - x[t])**2 + (y[t+lag_steps] - y[t])**2)/self.find_ind(ind).v0 ** 2
        # Calculate mean and standard deviation
        mean_autocorr = np.mean(msd)
        std_autocorr = np.std(msd)
        return mean_autocorr, std_autocorr

    # get mean squared displacement for all time lags up to t_lag_max
    def get_mean_squared_displacements_all(self, ind, t_lag_max):
        """
        Calculate autocorrelation for all time lags up to t_lag_max for a given individual and its trajectory.

        :param ind: Index of the individual.
        :type ind: int
        :param t_lag_max: Maximum length of time lag.
        :type t_lag_max: float
        :return: mean (float), standard deviation (float) of the autocorrelation.
        """
        # Convert t_lag into the number of time steps
        lag_steps_max = int(np.floor(t_lag_max / self.env_prop.time_step)+1)
        # Maximal number of sampled points
        num = min(100,lag_steps_max)
        # Initialize autocorrelation array
        mean = np.zeros(num+1)
        std = np.zeros(num+1)
        time_lags = np.zeros(num+1 )
        # Calculate autocorrelations for all valid time points
        for n in range(num+1):
            lag_steps = int(n * lag_steps_max / num)
            mean_msd, std_msd = self.find_mean_squared_displacement(ind, lag_steps * self.env_prop.time_step)
            mean[n] = mean_msd
            std[n] = std_msd
            time_lags[n] = lag_steps * self.env_prop.time_step
        return time_lags, mean, std

###
# PLOTTING FUNCTIONS
###
# Colours for each individual (maximum 10 at this moment), and each behaviour (maximum 10 at this moment)
ind_colors = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

# Functions that create a particular type of plot

# Plot snapshot of the action potential in the brain and current heading
def snapshot_brain(pattern, ind, heading, ax, **kwargs):
    """
    Generate a snapshot visualization of neural activity and heading orientation in allocentric coordinates.

    This function generates a polar plot showcasing brain activity for a specific
    individual index (`ind`) within the context of the given `pattern`. Additionally,
    it can also depict the heading direction if the `heading` parameter is specified.

    :param pattern: An object encapsulating the required data for generating
        the brain activity visualization.
    :type pattern: Pattern
    :param ind: The index of the individual for whom the brain activity is
        being visualized. This index is typically used to access a specific
        subset of data from the `pattern` object.
    :type ind: int
    :param heading: A flag indicating whether or not to visualize the individual's
        heading angle on the plot. If set to `True`, the directional heading
        is added to the generated polar plot.
    :type heading: bool
    :param ax: A matplotlib axes object on which the brain activity visualization
        will be rendered. This should be a polar axes to appropriately display
        the data.
    :type ax: matplotlib.projections.polar.PolarAxes
    :param kwargs: Additional keyword arguments to be passed to the `pcolormesh` plotting method.
    :type kwargs: dict

    :return: A colormap object representing the brain activity visualization.
        This object can be used for further manipulation or inclusion in
        other visualizations (matplotlib.collections.QuadMesh)
    """
    # prepare coordinates
    radius = np.array([0, 1])
    azimuth = np.linspace(0, 2*np.pi, pattern.env_prop.angle_num)
    r, th = np.meshgrid(radius, azimuth)
    # obtain and plot brain activity
    activity = pattern.nx[pattern.pt_now][ind][:pattern.env_prop.angle_num]
    if not pattern.find_ind(ind).allo:
        phi = pattern.nx[pattern.pt_now][ind][pattern.env_prop.angle_num+2] % (2 * np.pi)
        roll_angle = round(phi/pattern.env_prop.angle_step)
        activity = np.roll(activity,roll_angle)
    white_region = np.array([np.nan for i in range(pattern.env_prop.angle_num)])
    activity = np.array([white_region, activity])
    activity = activity.transpose()
    if 'vmin' not in kwargs:
        kwargs['vmin'] = 0
    if 'vmax' not in kwargs:
        kwargs['vmax'] = 1
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'inferno'
    if 'rasterized' not in kwargs:
        kwargs['rasterized'] = True
    cm = ax.pcolormesh(th, r, activity, **kwargs)
    ax.plot(azimuth, np.array([0.5 for i in range(pattern.env_prop.angle_num)]), color='black', linewidth=1)
    ax.set_yticks([])
    ax.set_xticks([])
    # obtain and plot heading angle
    if heading:
        vel_x,vel_y = pattern.find_velocity(ind)
        angle = np.arctan2(vel_y,vel_x)
        con = patches.ConnectionPatch([0, -0.5], [angle, 0.5], "data", "data",
                                      arrowstyle="-|>", mutation_scale=10, color=ind_colors[ind], fc=ind_colors[ind])
        ax.add_artist(con)
        ax.scatter(x=0, y=-0.5, color=ind_colors[ind], s=30)
    return cm

# Plot a snapshot of animal position and trajectory
def snapshot_move(pattern, heading, trajectory, labels, ax,  adjusted_plane=True):
    """
    Generates a visualization movement for all individuals in the 'pattern' object.
    In addition to the current individual and cue positions, heading direction vectors and trajectories can be plotted.
    The function sets up the plotting area, defines the grid boundaries, and adjusts its size dynamically.

    :param pattern: A data structure encapsulating the environmental and individual properties,
        including positions and cues.
    :type pattern: Pattern
    :param heading: A boolean flag indicating whether to display the heading direction as arrows
        for each individual.
    :type heading: bool
    :param trajectory: A boolean flag indicating whether to plot the trajectory of an individual's
        movement over time.
    :type trajectory: bool
    :param adjusted_plane: A boolean indicating whether the plot's boundaries should dynamically
        adjust to encapsulate the entire trajectory.
    :type adjusted_plane: bool
    :param labels: A boolean indicating whether to include axis labels ("space x" and "space y").
    :type labels: bool
    :param ax: A Matplotlib Axes object used to plot the environment, individuals, and trajectories.
    :type ax: matplotlib.axes.Axes

    :return: None
    """
    # set background colour
    ax.set_facecolor("whitesmoke")
    # find maximal and minimal targets
    min_pt = [pattern.env_prop.pos_max[0],pattern.env_prop.pos_max[1]]
    max_pt = [0,0]
    if adjusted_plane:
        for ind in range(pattern.env_prop.ind_num):
            for beh in pattern.ind_prop[ind].beh:
                for cue in beh.cue:
                    if cue[0] == 0 or cue[0] == 3:
                        for xy in range(2):
                            if cue[4+xy]>max_pt[xy]:
                                max_pt[xy] = cue[4+xy]
                            if cue[4+xy]<min_pt[xy]:
                                min_pt[xy] = cue[4+xy]
    # plot positions of each individual and its trajectory
    for ind in range(pattern.env_prop.ind_num):
        # stimuli
        for beh in pattern.ind_prop[ind].beh:
            for cue in beh.cue:
                if cue[0] == 0:
                    x,y = cue[4],cue[5]
                    radius = 2
                    circle = patches.Circle((x, y), radius, color="#c83737ff")  # (center_x, center_y), radius
                    ax.add_patch(circle)
                elif cue[0] == 3:
                    x,y = cue[4],cue[5]
                    radius = cue[2]
                    circle = patches.Circle((x, y), radius, color="#c83737ff")  # (center_x, center_y), radius
                    ax.add_patch(circle)
                elif cue[0] == 4:
                    x, y = cue[4], 0
                    radius = cue[2]
                    circle = patches.Circle((x, y), radius, color="#c83737ff")  # (center_x, center_y), radius
                    ax.add_patch(circle)
                    x, y = cue[5], cue[6]
                    radius = cue[2]
                    circle = patches.Circle((x, y), radius, color="#c83737ff")  # (center_x, center_y), radius
                    ax.add_patch(circle)
        # position
        x = pattern.nx[pattern.pt_now][ind][pattern.env_prop.angle_num]
        y = pattern.nx[pattern.pt_now][ind][pattern.env_prop.angle_num+1]
        ax.scatter(x=x, y=y, color=ind_colors[ind], s=15)
        # heading
        if heading:
            arr_l = min(pattern.env_prop.pos_max)/15
            phi = pattern.nx[pattern.pt_now][ind][pattern.env_prop.angle_num+2]
            arr_x, arr_y = np.cos(phi)*arr_l, np.sin(phi)*arr_l
            con = patches.ConnectionPatch([x, y], [x+arr_x, y+arr_y], "data", "data",
                                          arrowstyle="-|>", mutation_scale=10, color=ind_colors[ind], fc=ind_colors[ind])
            ax.add_artist(con)
        # trajectory
        if trajectory:
            traj_x = np.roll(pattern.nx[:, ind, pattern.env_prop.angle_num], -pattern.pt_now-1)
            traj_y = np.roll(pattern.nx[:, ind, pattern.env_prop.angle_num+1], -pattern.pt_now-1)
            ax.plot(traj_x, traj_y, color=ind_colors[ind], linewidth=0.7)
            if traj_x.min()<min_pt[0]:
                min_pt[0] = traj_x.min()
            if traj_x.max() > max_pt[0]:
                max_pt[0] = traj_x.max()
            if traj_y.min()<min_pt[1]:
                min_pt[1] = traj_y.min()
            if traj_y.max() > max_pt[1]:
                max_pt[1] = traj_y.max()
    # set limits and ticks
    if adjusted_plane:
        length = np.ceil(max(max_pt[0]-min_pt[0],max_pt[1]-min_pt[1])*1.4/10) * 10
        midpoints = [(max_pt[0]+min_pt[0])/2,(max_pt[1]+min_pt[1])/2]
        ax.set_xlim(midpoints[0]-length/2,midpoints[0]+length/2)
        ax.set_ylim(midpoints[1]-length / 2, midpoints[1] + length / 2)
        ax.set_xticks([midpoints[0]-length/2,midpoints[0]+length/2])
        ax.set_yticks([midpoints[1]-length /2, midpoints[1] + length / 2])
        ax.set_xticklabels([0,int(length)])
        ax.set_yticklabels([0,int(length)])
    else:
        ax.set_xlim([0, pattern.env_prop.pos_max[0]])
        ax.set_ylim([0, pattern.env_prop.pos_max[1]])
        # ax.set_yticks([])
        # ax.set_xticks([])
    # labels
    if labels:
        ax.set_xlabel("space $x$")
        ax.set_ylabel("space $y$")

# Plot a snapshot of brain trajectory
def snapshot_brain_traj(pattern, ind, beh_cmap, labels, ax, allocentric=True, S_crit=0.16, plot_S=False):
    """
    Plots the trajectory of brain activity for a given individual over a given period, segmented by periods of
    identical behavior, and visualized with color mapping based on behavior.

    :param pattern: The pattern object containing the trajectory data and associated properties.
    :type pattern: Pattern
    :param ind: Index of the individual whose trajectory is to be plotted.
    :type ind: int
    :param beh_cmap: Color map defining how different behaviors are visualized in terms of color filling in the trajectory plot.
    :type beh_cmap: list
    :param labels: Flag indicating whether axis labels should be included on the plot.
    :type labels: bool
    :param ax: The matplotlib.axes.Axes object on which the trajectory will be plotted.
    :type ax: matplotlib.axes.Axes
    :param allocentric: Determines whether the neural activity is shown in the allocentric (default, allocentric=True) or egocentric (allocentric=False) coordinates.
    :type allocentric: bool
    :param S_crit: The threshold value for the intrinsic stimulation to distinguish between systematic and random search in simulations with variable S.
    :type S_crit: float
    :param plot_S: Flag indicating whether the intrinsic stimulation value should be plotted on the trajectory plot.
    :type plot_S: bool
    """
    # get trajectory of brain between t_max-t_memory and t_max
    times, thetas, traj = pattern.get_brain_trajectory(ind,allocentric)
    # put theta=0 in the middle and modify the axis for angles accordingly (+ add a point)
    traj = np.roll(traj, -pattern.env_prop.angle_num//2, axis=1)
    thetas= np.asarray([-np.pi]+[-np.pi+(1+2*k)*np.pi/pattern.env_prop.angle_num for k in range(pattern.env_prop.angle_num)]+[np.pi])
    # cut trajectory to intervals of identical behaviour + plot these trajectories with appropriate colour
    switch_times, switch_t_index, switch_behaviour = pattern.get_behaviour_switch(ind, S_crit)
    for i in range(len(switch_t_index)-1):
        tim_i_min = switch_t_index[i]
        tim_i_max = switch_t_index[i+1]
        plot_traj = np.transpose(traj[tim_i_min:tim_i_max, :])
        plot_traj = np.vstack([plot_traj, plot_traj[0,:]])
        time = np.linspace(times[tim_i_min],times[tim_i_max-1],num=tim_i_max-tim_i_min+1)
        ax.pcolormesh(time, thetas, plot_traj, cmap=beh_cmap[switch_behaviour[i]], vmin=0, vmax=1, rasterized=True)
    # set limits
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-np.pi, np.pi)
    # set ticks
    ax.set_yticks([-np.pi,0,np.pi])
    ax.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax.set_xticks([pattern.env_prop.time_max-(pattern.time_num-2)*pattern.env_prop.time_step, pattern.env_prop.time_max])
    # labels
    if labels:
        ax.set_xlabel("time t")
        ax.set_ylabel("angle $\\theta$")
    # plot S
    if plot_S:
        ax_s = ax.twinx()
        times, s_vals = pattern.get_S_trajectory(ind)
        ax_s.plot(times, s_vals, color="black", linewidth=1)
        ax_s.set_ylim(-0.1,1.1)
        ax_s.set_ylabel("stimulation $S$")

# Plot statistics of an animal trajectory
def statistics_move(pattern, ind, t_lag_max, labels, ax):
    """
    Generates a plot of statistics for the animal's movement (autocorrelation and mean squared displacement).

    :param pattern: A data structure encapsulating the environmental and individual properties,
        including positions and cues.
    :type pattern: Pattern
    :param ind: Index of the individual whose statistics are to be plotted.
    :type ind: int
    :param t_lag_max: Maximum length of time lag for the plotted statistics.
    :type t_lag_max: float
    :param labels: A boolean indicating whether to include axis labels ("space x" and "space y").
    :type labels: bool
    :param ax: A list of Matplotlib Axes object used to plot each type of statistics.
    :type ax: list[matplotlib.axes.Axes]

    :return: None
    """
    # get statistics to be plotted
    lag_auto, mean_auto, std_auto = pattern.get_autocorrelations_all(ind,t_lag_max)
    lag_msd, mean_msd, std_msd = pattern.get_mean_squared_displacements_all(ind,t_lag_max)

    # plot mean squared displacement (MSD) in black
    ax[0].plot(lag_msd, mean_msd, color='black', label="Mean")
    ax[0].fill_between(lag_msd, mean_msd - std_msd, mean_msd + std_msd, color='black', alpha=0.2, edgecolor='none', label="Std Dev")

    # plot autocorrelation in red
    ax[1].plot(lag_auto, mean_auto, color='tab:red', label="Mean")
    ax[1].fill_between(lag_auto, mean_auto - std_auto, mean_auto + std_auto, color='tab:red', alpha=0.2, edgecolor='none', label="Std Dev")

    # set limits
    ax[0].set_xlim(0, t_lag_max)
    ax[1].set_xlim(0, t_lag_max)
    #ax[0].set_ylim([0, (t_lag_max) ** 2])
    ax[0].set_ylim([0, np.max(mean_msd+std_msd)])
    ax[1].set_ylim([-1.1, 1.1])
    ax[1].axhline(0, color='tab:red', linestyle=":",linewidth=1,zorder=1)
    ax[1].axhline(1, color='tab:red', linestyle=":", linewidth=1, zorder=1)
    ax[1].axhline(-1, color='tab:red', linestyle=":", linewidth=1, zorder=1)

    # labels
    if labels:
        ax[1].tick_params(axis="y", colors="tab:red")
        ax[1].yaxis.label.set_color("tab:red")
        ax[0].set_xlabel("time lag $\\Delta t$")
        ax[0].set_ylabel("mean squared displacement")
        ax[0].set_xlabel("time lag $\\Delta t$")
        ax[1].set_ylabel("autocorrelation")

# Plot a snapshot of brain and movement trajectories
def snapshot_trajectory(env_prop, ind_prop, beh_cmap, fig_name, labels, seed, deterministic, allocentric=True, adjusted_plane=True, ticks=True, stat_t_lag=None, S_crit=0.16, plot_S=False):
    """
    Simulates a trajectory snapshot of an environment with specified properties,
    and generates a detailed figure depicting both the movement within the
    environment and the corresponding brain activity for each individual.
    Saves the generated figure and simulation properties to disk.

    :param env_prop: The properties of the environment being simulated.
    :type env_prop: EnvProp
    :param ind_prop: The properties of the individuals within the environment.
    :type ind_prop: list[IndProp]
    :param beh_cmap: The colormap used for visualizing brain activities.
    :type beh_cmap: matplotlib.colors.Colormap
    :param fig_name: The base name of the figure file to be saved.
    :type fig_name: str
    :param labels: A boolean indicator if labels are to be plotted.
    :type labels: bool
    :param seed: Random seed for reproducibility of stochastic simulations.
    :type seed: int or None
    :param deterministic: Determines whether the simulation should run in a deterministic or stochastic mode.
    :type deterministic: bool
    :param allocentric: Specifies if the neural dynamics is plotted in allocentric or egocentric coordinates.
    :type allocentric: bool
    :param adjusted_plane: Adjusts the visualization plane for movement plotting.
        Defaults to True.
    :type adjusted_plane: bool
    :param ticks: Determines whether axis ticks are to be plotted. Defaults to True.
    :type ticks: bool
    :param stat_t_lag: Determines the length of time lag for the plotted statistics. Defaults to None.
    :type stat_t_lag: float or None
    :param S_crit: The threshold value for the intrinsic stimulation to distinguish between systematic and random search in simulations with variable S.
    :type S_crit: float
    :param plot_S: Flag indicating whether the intrinsic stimulation value should be plotted on the trajectory plot.
    :type plot_S: bool
    """

    # create a simulation and save its properties
    if seed is not None:
        np.random.seed(seed)
    pattern = Pattern(env_prop, ind_prop)
    pattern.save_properties("Figures//Metadata//",fig_name)
    # create figure
    cell_size = 2.5
    fig = plt.figure(figsize=(cell_size*(0.15*plot_S+1), cell_size*(1+pattern.env_prop.ind_num/2)), layout="constrained")
    # crate axes
    gs = fig.add_gridspec(2+pattern.env_prop.ind_num, 1)
    ax_move = fig.add_subplot(gs[pattern.env_prop.ind_num:])
    ax_brain = [fig.add_subplot(gs[ind]) for ind in range(pattern.env_prop.ind_num)]
    # do the simulation
    pattern.simulate(deterministic)
    # do the plots
    # plot movement
    snapshot_move(pattern, False, True, labels, ax_move, adjusted_plane)
    # plot the brains
    for ind in range(pattern.env_prop.ind_num):
        snapshot_brain_traj(pattern, ind, beh_cmap, labels, ax_brain[ind], allocentric, S_crit, plot_S)
    # remove ticks if necessary
    if not ticks:
        ax_move.set_xticklabels([])
        ax_move.set_yticklabels([])
        for ind in range(pattern.env_prop.ind_num):
            ax_brain[ind].set_xticklabels([])
    # save the figure
    if deterministic:
        fig_name += "_Det"
    else:
        fig_name += "_Stoch"
    fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
    fig.savefig("Figures//" + fig_name + ".svg")
    # plot statistics if necessary
    if stat_t_lag is not None:
        fig_stats, ax_msd = plt.subplots(ncols=pattern.env_prop.ind_num,figsize=(cell_size, cell_size*pattern.env_prop.ind_num))
        if pattern.env_prop.ind_num == 1:
            ax_msd = [ax_msd]
        for ind in range(pattern.env_prop.ind_num):
            ax_auto = ax_msd[ind].twinx()
            statistics_move(pattern, 0, stat_t_lag, labels, [ax_msd[ind],ax_auto])
        fig_stats.savefig("Figures//" + fig_name + "_Stats.png", dpi=300, format='png')
        fig_stats.savefig("Figures//" + fig_name + "_Stats.svg")

# Create movie snapshots of both brain and movement dynamics
def movie_snapshots(env_prop, ind_prop, plot_time, heading, trajectory, anim_name, seed):
    """
    Generates a series of snapshots for a simulation.

    This function handles the creation of a simulation visualization over a specified time period,
    saving each snapshot to a designated folder. It creates visual representations of environment
    movement and individual neural activities in a polar grid. The resulting snapshots can be used as frames for a movie.

    :param env_prop: Properties of the environment.
    :type env_prop: EnvProp
    :param ind_prop: Properties of the individuals.
    :type ind_prop: list[IndProp]
    :param plot_time: Interval of time between simulation snapshots, determining the granularity of captured frames.
    :type plot_time: float
    :param heading: A boolean flag indicating whether to include individual heading vectors in the snapshots.
    :type heading: str
    :param trajectory: A boolean flag indicating whether to include individual trajectories in the snapshots.
    :type trajectory: bool
    :param anim_name: The base name of the animation, used for naming the folder where snapshots are stored
        and for naming each snapshot image.
    :type anim_name: str
    :param seed: Seed value for initializing random number generation. If None, no specific seed is used.
    :type seed: int or None
    """

    # create figure
    row = 5
    cell_size = 1.5
    if env_prop.ind_num > 2:
        row += 2 * ((env_prop.ind_num - 3) // 4 + 1)
    fig = plt.figure(figsize=(4 * cell_size, row / 2 * cell_size), layout="constrained")
    # crate axes
    gs = fig.add_gridspec(row, 4)
    # - colorbar
    gs_col = gs[0,0].subgridspec(nrows=2, ncols=3, width_ratios=[0.2,1,0.2],height_ratios=[1,1])
    ax_col = fig.add_subplot(gs_col[0, 1])
    ax_col.set_yticks([])
    ax_col.set_xticks([])
    # - move
    ax_move = fig.add_subplot(gs[0:5, 1:4])
    # - brain
    ax_brain = []
    ax_brain.append(fig.add_subplot(gs[1:3, 0], projection='polar'))
    if env_prop.ind_num > 1:
        ax_brain.append(fig.add_subplot(gs[3:5, 0], projection='polar'))
    if env_prop.ind_num > 2:
        for extra_ind in range(env_prop.ind_num - 2):
            rw = 2 * (extra_ind // 4) + 5
            cl = extra_ind % 4
            ax_brain.append(fig.add_subplot(gs[rw:rw + 2, cl], projection='polar'))
    # prepare folder
    date_time = datetime.datetime.now()
    folder_specifier = "_" + str(date_time.date().day) + "_" + str(date_time.date().month) + "_" + str(
        date_time.date().year) + "_" + str(date_time.time().hour) + "_" + str(date_time.time().minute)
    folder_name = "Movies\\" + anim_name + folder_specifier
    os.mkdir(folder_name)
    # create a simulation and save its properties
    if seed is not None:
        np.random.seed(seed)
    pattern = Pattern(env_prop, ind_prop)
    pattern.save_properties(folder_name)
    # do the simulation, plot and save snapshots
    plot_index = 0
    while pattern.time <= pattern.env_prop.time_max:
        if pattern.time >= plot_index*plot_time:
            # snapshot of movement
            snapshot_move(pattern, heading, trajectory, False, False, ax_move)
            # snapshot of the brain
            for ind in range(pattern.env_prop.ind_num):
                cm = snapshot_brain(pattern,ind,True,ax_brain[ind])
            # create the colorbar if this is the first simulation
            if plot_index == 0:
                fig.colorbar(cm, cax=ax_col, orientation='horizontal', ticks=[],)
                ax_col.set_xlabel("neural activity", labelpad=0)
                xmin, xmax = ax_col.get_xlim()
                ax_col.set_xticks([xmin,xmax])
                ax_col.set_xticklabels(["low", "high"])
            # update simulation
            pattern.update_pattern()
            # take a snapshot and clear the canvas
            fig.savefig(folder_name+"\\"+anim_name+"-"+str(plot_index+1)+".png", dpi=300, format='png')  # save the figure to file
            for ax in ax_brain:
                for artist in ax.lines + ax.collections + ax.patches:
                    artist.remove()
            for artist in ax_move.lines + ax_move.collections + ax_move.patches:
                artist.remove()
            # report creation of the frame
            print("Created frame: " + str(plot_index))
            plot_index += 1
        else:
            # update simulation
            pattern.update_pattern()

# Create snapshots of movements towards targets
def snapshot_targets(env_prop, ind_prop, beh_cmap, fig_name, labels, seed, deterministic, allocentric=True, adjusted_plane=True, repeats=10):
    """
    Simulates many trajectories of animals that move towards target, until they reach the first target or time hits the end of the simulation.

    :param env_prop: The properties of the environment being simulated.
    :type env_prop: EnvProp
    :param ind_prop: The properties of the individuals within the environment.
    :type ind_prop: list[IndProp]
    :param beh_cmap: The colormap used for visualizing brain activities.
    :type beh_cmap: matplotlib.colors.Colormap
    :param fig_name: The base name of the figure file to be saved.
    :type fig_name: str
    :param labels: A boolean indicator if labels are to be plotted.
    :type labels: bool
    :param seed: Random seed for reproducibility of stochastic simulations.
    :type seed: int or None
    :param deterministic: Determines whether the simulation should run in a deterministic or stochastic mode.
    :type deterministic: bool
    :param allocentric: Specifies if the neural dynamics is plotted in allocentric or egocentric coordinates.
    :type allocentric: bool
    :param adjusted_plane: Adjusts the visualization plane for movement plotting.
        Defaults to True.
    :type adjusted_plane: bool
    :param repeats: Number of times the simulation is repeated. Defaults to 10.
    :type repeats: int
    """
    # create the initial seed
    if seed is not None:
        np.random.seed(seed)
    # create figure
    cell_size = 2.5
    fig = plt.figure(figsize=(cell_size, cell_size*(1+env_prop.ind_num/2)), layout="constrained")
    # crate axes
    gs = fig.add_gridspec(2+env_prop.ind_num, 1)
    ax_move = fig.add_subplot(gs[env_prop.ind_num:])
    ax_brain = [fig.add_subplot(gs[ind]) for ind in range(env_prop.ind_num)]
    for rep in range(repeats):
        print("Repeat: " + str(rep))
        if rep == 0:
            # create a simulation and save its properties
            pattern = Pattern(env_prop, ind_prop)
            pattern.save_properties("Figures//Metadata//",fig_name)
            # do the simulation
            pattern.simulate(deterministic)
            # do the plots
            # plot movement
            snapshot_move(pattern, False, True, labels, ax_move, adjusted_plane)
            # plot the brains
            for ind in range(pattern.env_prop.ind_num):
                snapshot_brain_traj(pattern, ind, beh_cmap, labels, ax_brain[ind], allocentric)
            # prolong maximal time and memory time
            env_prop.time_max *= 3
            env_prop.time_memory *= 3
        else:
            # create a simulation
            pattern = Pattern(env_prop, ind_prop)
            # do the simulation
            percentage = 0
            while pattern.time <= pattern.env_prop.time_max and pattern.get_minimal_cue_distance() > 2:
                if pattern.time >= percentage * pattern.env_prop.time_max / 100:
                    print("Simulation finished at least up to" + str(percentage) + "%")
                    percentage += 1
                pattern.update_pattern(deterministic)
            # do the plots
            for ind in range(pattern.env_prop.ind_num):
                traj_x = np.roll(pattern.nx[:, ind, pattern.env_prop.angle_num], -pattern.pt_now-1)
                traj_y = np.roll(pattern.nx[:, ind, pattern.env_prop.angle_num+1], -pattern.pt_now-1)
                ax_move.plot(traj_x, traj_y, color="lightgray", linewidth=0.7,zorder=0)
    # save the figure
    if deterministic:
        fig_name += "_Det"
    else:
        fig_name += "_Stoch"
    fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
    fig.savefig("Figures//" + fig_name + ".svg")

# Create a plot of the alpha_1 coefficient as a function of various parameters of the model
def plot_alpha_dependence(kernel, S, T):
    """
    Create a plot that shows how the excitation number alpha depends on various parameters of the model.
    The input 'beh_prop' must have a single cue of type=2 and kappa=0.

    :param kernel: Default kernel of the neural interactions (same as in BehProp).
    :type kernel: Union(tuple,list)
    :param S: Default intrinsic stimulation (same as in BehProp).
    :type S: float
    :param T: Default neural noise (same as in BehProp).
    :type T: float
    """
    # prepare plot
    fig = plt.figure(figsize=(7.5, 2.5), layout="constrained")
    # create axes
    gs = fig.add_gridspec(1, 3)
    ax_J = fig.add_subplot(gs[1])
    ax_s = fig.add_subplot(gs[0])
    ax_T = fig.add_subplot(gs[2])
    # prepare plotting data
    num = 150
    J_max = 10
    T_max = 10
    Js = np.linspace(0, J_max, num=num)
    ss = np.linspace(0, 1, num=num)
    Ts = np.linspace(0, T_max, num=num)
    alpha_J = np.zeros(num)
    alpha_s = np.zeros(num)
    alpha_T = np.zeros(num)
    # set s0 to 0.5 to have plots that visually nicely fit into range 0,2
    for val in range(num):
        # get value for alpha_J
        kernel_val = kernel.copy()
        kernel_val[1] = Js[val]
        beh_prop = BehProp(kernel=kernel_val, T=T, S=S)
        alpha_J[val] = beh_prop.alpha(1)
        # get value for alpha_s
        beh_prop = BehProp(kernel=kernel, T=T, S=ss[val])
        alpha_s[val] = beh_prop.alpha(1)
        # get value for alpha_T
        beh_prop = BehProp(kernel=kernel, T=Ts[val], S=S)
        alpha_T[val] = beh_prop.alpha(1)
    # make plot
    ax_J.fill_between([0, J_max], [1, 1], color="#FDE27D")
    ax_s.fill_between([0, 1], [1, 1], color="#FDE27D")
    ax_T.fill_between([0, T_max], [1, 1], color="#FDE27D")
    ax_J.fill_between([0, J_max], [1, 1], [1000, 1000], color="#EC7063")
    ax_s.fill_between([0, 1], [1, 1], [1000, 1000], color="#EC7063")
    ax_T.fill_between([0, T_max], [1, 1], [1000, 1000], color="#EC7063")
    ax_J.plot(Js, alpha_J, color='black')
    ax_s.plot(ss, alpha_s, color='black')
    ax_T.plot(Ts, alpha_T, color='black')
    ax_J.axhline(y=1, color='black', linestyle=':')
    ax_s.axhline(y=1, color='black', linestyle=':')
    ax_T.axhline(y=1, color='black', linestyle=':')
    # labels
    ax_J.set_xlabel("interaction strength $J$")
    ax_s.set_xlabel("intrinsic stimulation $S$")
    ax_T.set_xlabel("neural noise $T$")
    #ax_J.set_ylabel("excitation $\\alpha$")
    #ax_s.set_ylabel("excitation $\\alpha$")
    #ax_eta.set_ylabel("excitation $\\alpha$")
    ax_s.set_ylabel("excitation $\\alpha$")
    # limits
    ax_J.set_xlim([0, J_max])
    ax_s.set_xlim([0, 1])
    ax_T.set_xlim([0, T_max])
    ax_J.set_ylim([0, 2])
    ax_s.set_ylim([0, 2])
    ax_T.set_ylim([0, 2])
    # ticks
    ax_T.set_yticks([])
    ax_J.set_yticks([])
    ax_s.set_yticks([0, 1, 2])
    # show plot
    fig.savefig("Figures/Fig_AlphaDependence.svg")
    fig.savefig("Figures/Fig_AlphaDependence.png")

# Create a plot of the phase transtion between excited and inhibited phase of neural activity
def plot_exc_inh_phase_heatmap(env_prop,fig_name,seed=None,beh_prop=None):
    """
    Plots a heatmap portraying the phase transition between excited and inhibited phase of neural activity.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig = plt.figure(figsize=(2.5, 2.5), layout="constrained")
    gs = fig.add_gridspec(2, 1)
    ax_stim = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    # create axes
    num = 100
    stim = np.linspace(0,1,num+1)
    angles = np.asarray([-np.pi]+[-np.pi+(1+2*k)*np.pi/env_prop.angle_num for k in range(env_prop.angle_num)]+[np.pi])
    stim_plot = np.zeros((num,env_prop.angle_num+1))
    phase_plot = np.zeros((num,env_prop.angle_num+1))
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st in range(num):
        # create pattern
        if beh_prop is None:
            beh_prop_var = BehProp(S=(stim[st+1]+stim[st])/2)
        else:
            beh_prop['S']=(stim[st+1]+stim[st])/2
            beh_prop_var = BehProp(beh_prop)
        beh_prop_var.cue = [[2,False,0,1,np.pi/2]]
        beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2*beh_prop_var.brain_fp, np.pi / 4, -np.pi/2]
        ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
        pattern = Pattern(env_prop,ind_prop)
        # simulate the pattern
        pattern.simulate(True)
        # obtain neural pattern
        traj = np.roll(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num], pattern.env_prop.angle_num//2)
        traj = np.append(traj, traj[0])
        phase_plot[st] = traj
        # obtain the stimulation profile
        for ang in range(pattern.env_prop.angle_num):
            stim_value = pattern.find_stimulation(0,ang)*(stim[st+1]+stim[st])/2
            stim_plot[st][(ang+pattern.env_prop.angle_num//2)%pattern.env_prop.angle_num] = stim_value
            if ang == 0:
                stim_plot[st][-1] = stim_value
        print("Finished step: "+str(st)+"/"+str(num))
    # find the phase transition boundary
    st_boundary = 0
    while np.sum(phase_plot[st_boundary])<np.pi/8/env_prop.angle_step:
        st_boundary += 1
    # Create own colour map
    yellows = LinearSegmentedColormap.from_list("Yellows", [(1,1,0.8), (0.8, 0.6, 0)])
    # make the plot
    stim_plot = stim_plot.transpose()
    phase_plot = phase_plot.transpose()
    ax_stim.pcolormesh(stim, angles, stim_plot, cmap="Greys", vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(stim[:st_boundary+1], angles, phase_plot[:,:st_boundary], cmap=yellows, vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(stim[st_boundary:], angles, phase_plot[:,st_boundary:], cmap="Reds", vmin=0, vmax=1, rasterized=True)
    # plot boundary and heading lines
    #ax_stim.axvline(st_boundary/num,-np.pi,np.pi, color="black", linestyle="--")
    ax_phase.axvline(st_boundary/num, -np.pi, np.pi, color="black", linestyle="--")
    ax_stim.axhline(-np.pi/2, 0, 1, color="black", linestyle=":")
    #ax_phase.axhline(0, 0, 1, color="black", linestyle=":")
    # set limits
    ax_stim.set_xlim([0,1])
    ax_stim.set_ylim([-np.pi, np.pi])
    ax_phase.set_xlim([0, 1])
    ax_phase.set_ylim([-np.pi, np.pi])
    # set ticks
    ax_stim.set_yticks([-np.pi,0,np.pi])
    ax_phase.set_yticks([-np.pi, 0, np.pi])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_phase.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_xticks([])
    ax_phase.set_xticks([0, 1])
    # labels
    ax_stim.set_ylabel("angles $\\theta$")
    ax_phase.set_ylabel("angles $\\theta$")
    ax_phase.set_xlabel("stimulation strength $s_0$")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase transtion between excited and induced phase of neural activity
def plot_exc_ind_phase_heatmap(env_prop,kappa_max, fig_name, seed=None,beh_prop=None):
    """
    Plots a heatmap portraying the phase transition between excited and induced phase of neural activity.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param kappa_max: The maximum value of stimulation heterogeneity.
    :type kappa_max: float
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig = plt.figure(figsize=(2.5, 2.5), layout="constrained")
    gs = fig.add_gridspec(2, 1)
    ax_stim = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    # create axes
    num = 100
    kappa = np.linspace(0,kappa_max,num+1)
    angles = np.asarray([-np.pi]+[-np.pi+(1+2*k)*np.pi/env_prop.angle_num for k in range(env_prop.angle_num)]+[np.pi])
    stim_plot = np.zeros((num,env_prop.angle_num+1))
    phase_plot = np.zeros((num,env_prop.angle_num+1))
    # prepare the variable for finding the phase transition boundary
    st_boundary = 0
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st in range(num):
        # create pattern
        if beh_prop is None:
            beh_prop_var = BehProp()
        else:
            beh_prop_var = BehProp(beh_prop)
        beh_prop_var.cue = [[2,False,kappa[st],1,np.pi/2]]
        beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2*beh_prop_var.brain_fp, np.pi / 4, -np.pi/2]
        ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
        pattern = Pattern(env_prop,ind_prop)
        # simulate the pattern
        pattern.simulate(True)
        # obtain neural pattern
        traj = np.roll(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num], pattern.env_prop.angle_num//2)
        traj = np.append(traj, traj[0])
        phase_plot[st] = traj
        # obtain the stimulation profile
        for ang in range(pattern.env_prop.angle_num):
            stim_value = pattern.find_stimulation(0,ang)
            stim_plot[st][(ang+pattern.env_prop.angle_num//2)%pattern.env_prop.angle_num] = stim_value
            if ang == 0:
                stim_plot[st][-1] = stim_value
        # update the boundary
        angle = np.linspace(0, 2 * np.pi-env_prop.angle_step, env_prop.angle_num)
        heading = np.sum(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num]*np.sin(angle)) # intended heading - y component
        if heading < 0:
            st_boundary += 1
        print("Finished step: "+str(st)+"/"+str(num))
    # make the plot
    stim_plot = stim_plot.transpose()
    phase_plot = phase_plot.transpose()
    ax_stim.pcolormesh(kappa, angles, stim_plot, cmap="Greys", vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(kappa[:st_boundary+1], angles, phase_plot[:,:st_boundary], cmap="Reds", vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(kappa[st_boundary:], angles, phase_plot[:,st_boundary:], cmap="Blues", vmin=0, vmax=1, rasterized=True)
    # plot boundary and heading lines
    ax_phase.axvline(kappa_max*st_boundary/num, -np.pi, np.pi, color="black", linestyle="--")
    ax_stim.axhline(-np.pi/2, 0, 1, color="black", linestyle=":")
    #ax_phase.axhline(0, 0, 1, color="black", linestyle=":")
    # set limits
    ax_stim.set_xlim([0,kappa_max])
    ax_stim.set_ylim([-np.pi, np.pi])
    ax_phase.set_xlim([0, kappa_max])
    ax_phase.set_ylim([-np.pi, np.pi])
    # set ticks
    ax_stim.set_yticks([-np.pi,0,np.pi])
    ax_phase.set_yticks([-np.pi, 0, np.pi])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_phase.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_xticks([])
    ax_phase.set_xticks([0, kappa_max])
    # labels
    ax_stim.set_ylabel("angles $\\theta$")
    ax_phase.set_ylabel("angles $\\theta$")
    ax_phase.set_xlabel("stimulation heterogeneity $\\kappa$")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase transtion between induced and inhibited phase of neural activity
def plot_inh_ind_phase_heatmap(env_prop,kappa_max,fig_name, seed=None,beh_prop=None):
    """
    Plots a heatmap portraying the phase transition between induced and inhibited phase of neural activity.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param kappa_max: The maximum value of stimulation heterogeneity.
    :type kappa_max: float
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig = plt.figure(figsize=(2.5, 2.5), layout="constrained")
    gs = fig.add_gridspec(2, 1)
    ax_stim = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    # create axes
    num = 100
    stim = np.linspace(0,1,num+1)
    angles = np.asarray([-np.pi]+[-np.pi+(1+2*k)*np.pi/env_prop.angle_num for k in range(env_prop.angle_num)]+[np.pi])
    stim_plot = np.zeros((num,env_prop.angle_num+1))
    phase_plot = np.zeros((num,env_prop.angle_num+1))
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st in range(num):
        # create pattern
        if beh_prop is None:
            beh_prop_var = BehProp(S=(stim[st+1]+stim[st])/2)
        else:
            beh_prop['S']=(stim[st+1]+stim[st])/2
            beh_prop_var = BehProp(beh_prop)
        beh_prop_var.cue = [[2,False,kappa_max,1,np.pi/2]]
        beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2*beh_prop_var.brain_fp, np.pi / 4, -np.pi/2]
        ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
        pattern = Pattern(env_prop,ind_prop)
        # simulate the pattern
        pattern.simulate(True)
        # obtain neural pattern
        traj = np.roll(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num], pattern.env_prop.angle_num//2)
        traj = np.append(traj, traj[0])
        phase_plot[st] = traj
        # obtain the stimulation profile
        for ang in range(pattern.env_prop.angle_num):
            stim_value = pattern.find_stimulation(0,ang)*(stim[st+1]+stim[st])/2
            stim_plot[st][(ang+pattern.env_prop.angle_num//2)%pattern.env_prop.angle_num] = stim_value
            if ang == 0:
                stim_plot[st][-1] = stim_value
        print("Finished step: "+str(st)+"/"+str(num))
    # find the phase transition boundary
    st_boundary = 0
    while np.sum(phase_plot[st_boundary])<np.pi/8/env_prop.angle_step:
        st_boundary += 1
    # Create own colour map
    yellows = LinearSegmentedColormap.from_list("Yellows", [(1,1,0.8), (0.8, 0.6, 0)])
    # make the plot
    stim_plot = stim_plot.transpose()
    phase_plot = phase_plot.transpose()
    ax_stim.pcolormesh(stim, angles, stim_plot, cmap="Greys", vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(stim[:st_boundary+1], angles, phase_plot[:,:st_boundary], cmap=yellows, vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(stim[st_boundary:], angles, phase_plot[:,st_boundary:], cmap="Blues", vmin=0, vmax=1, rasterized=True)
    # plot boundary and heading lines
    #ax_stim.axvline(st_boundary/num,-np.pi,np.pi, color="black", linestyle="--")
    ax_phase.axvline(st_boundary/num, -np.pi, np.pi, color="black", linestyle="--")
    ax_stim.axhline(-np.pi/2, 0, 1, color="black", linestyle=":")
    #ax_phase.axhline(0, 0, 1, color="black", linestyle=":")
    # set limits
    ax_stim.set_xlim([0,1])
    ax_stim.set_ylim([-np.pi, np.pi])
    ax_phase.set_xlim([0, 1])
    ax_phase.set_ylim([-np.pi, np.pi])
    # set ticks
    ax_stim.set_yticks([-np.pi,0,np.pi])
    ax_phase.set_yticks([-np.pi, 0, np.pi])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_phase.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_xticks([])
    ax_phase.set_xticks([0, 1])
    # labels
    ax_stim.set_ylabel("angles $\\theta$")
    ax_phase.set_ylabel("angles $\\theta$")
    ax_phase.set_xlabel("stimulation strength $s_0$")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase transtion within induced phase of neural activity
def plot_ind_phase_heatmap(env_prop, beh_prop, s, kappa, fig_name, seed=None):
    """
    Plot decision-making between two targets as their angular difference is varied.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: BehProp object describing the behavioral properties of the individual.
    :type beh_prop: BehProp
    :param s: strength of cues
    :type s: list[float]
    :param kappa: heterogeneity of cues
    :type kappa: list[float]
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig = plt.figure(figsize=(2.5, 2.5), layout="constrained")
    gs = fig.add_gridspec(2, 1)
    ax_stim = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    # create axes
    num = 100
    ang_diff = np.linspace(0,np.pi,num+1)
    angles = np.asarray([-np.pi]+[-np.pi+(1+2*k)*np.pi/env_prop.angle_num for k in range(env_prop.angle_num)]+[np.pi])
    stim_plot = np.zeros((num,env_prop.angle_num+1))
    phase_plot = np.zeros((num,env_prop.angle_num+1))
    # hold the animal in a fixed position for the duration of the experiment, fix initial heading
    beh_prop.init_hold = env_prop.time_max
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st in range(num):
        # create pattern
        beh_prop_var = copy.deepcopy(beh_prop)
        ang_diff_val = float((ang_diff[st + 1] + ang_diff[st]) / 2) / 2
        beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2*beh_prop_var.brain_fp, np.pi / 4, 0]
        beh_prop_var.cue = [[2, False, kappa[0],s[0], -ang_diff_val],[2, False, kappa[1],s[1], ang_diff_val]]
        ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
        pattern = Pattern(env_prop,ind_prop)
        # simulate the pattern
        pattern.simulate(True)
        # obtain neural pattern
        traj = np.roll(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num], pattern.env_prop.angle_num//2)
        traj = np.append(traj, traj[0])
        phase_plot[st] = traj
        # obtain the stimulation profile
        for ang in range(pattern.env_prop.angle_num):
            stim_value = pattern.find_stimulation(0,ang)/pattern.find_ind(0).T
            stim_plot[st][(ang+pattern.env_prop.angle_num//2)%pattern.env_prop.angle_num] = stim_value
            if ang == 0:
                stim_plot[st][-1] = stim_value
        print("Finished step: " + str(st) + "/" + str(num))
    # make the plot
    stim_plot = stim_plot.transpose()
    phase_plot = phase_plot.transpose()
    ax_stim.pcolormesh(ang_diff, angles, stim_plot, cmap="Greys", vmin=0, vmax=1, rasterized=True)
    ax_phase.pcolormesh(ang_diff, angles, phase_plot, cmap="Blues", vmin=0, vmax=1, rasterized=True)
    # add the lines describing the position of the cues
    ax_phase.plot(ang_diff, ang_diff/2, color="black", linestyle="--")
    ax_phase.plot(ang_diff, -ang_diff/2, color="black", linestyle="--")
    # set limits
    ax_stim.set_xlim([0,np.pi])
    ax_stim.set_ylim([-np.pi, np.pi])
    ax_phase.set_xlim([0, np.pi])
    ax_phase.set_ylim([-np.pi, np.pi])
    # set ticks
    ax_stim.set_yticks([-np.pi,0,np.pi])
    ax_phase.set_yticks([-np.pi, 0, np.pi])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_phase.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_yticklabels(["-$\\pi$", 0, "$\\pi$"])
    ax_stim.set_xticks([])
    ax_phase.set_xticks([0, np.pi])
    ax_phase.set_xticklabels([0, "$\\pi$"])
    # labels
    ax_stim.set_ylabel("angles $\\theta$")
    ax_phase.set_ylabel("angles $\\theta$")
    ax_phase.set_xlabel("angular difference")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase diagram for the neural activity
# Input: env_prop, beh_prop, fig_name
def plot_phases_heatmap(env_prop, kappa_max, fig_name, seed=None, beh_prop=None):
    """
    Plot the phase diagram for the neural activity.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param kappa_max: Maximal value of the stimulation heterogeneity.
    :type kappa_max: float
    :param fig_name: Name of the figure.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig = plt.figure(figsize=(3.2, 3), layout="constrained")
    gs = fig.add_gridspec(1, 1)
    ax_phase = fig.add_subplot(gs[0])
    # create axes
    num = 50
    kappa = np.linspace(0,kappa_max,num+1)
    stim = np.linspace(0, 1, num + 1)
    phase_plot = np.zeros((num,num))
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st0 in range(num):
        for st1 in range(num):
            # get stimulation values
            print("Generated point:"+str(st0)+", "+str(st1))
            # create pattern
            if beh_prop is None:
                beh_prop_var = BehProp(S=(stim[st1+1]+stim[st1])/2)
            else:
                beh_prop['S'] = (stim[st1 + 1] + stim[st1]) / 2
                beh_prop_var = BehProp(beh_prop)
            beh_prop_var.cue = [[2, False, kappa[st0], 1, np.pi / 2]]
            beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2 * beh_prop_var.brain_fp, np.pi / 4, -np.pi / 2]
            ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
            pattern = Pattern(env_prop, ind_prop)
            # simulate the pattern
            pattern.simulate(True)
            # classify the phase
            # 0 = check if you are in the inhibited phase
            if np.sum(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num]) < np.pi / 8 / env_prop.angle_step:
                phase_plot[st0][st1] = 0
            else:
                angles = np.linspace(0, 2 * np.pi-env_prop.angle_step, env_prop.angle_num)
                heading = np.sum(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num]*np.sin(angles)) # intended heading - y component
                # 1 = check if you are in the excited phase
                if heading < 0:
                    phase_plot[st0][st1] = 1
                # 1 = check if you are in the induced phase
                else:
                    phase_plot[st0][st1] = 2
    # make the plot
    phase_plot = phase_plot.transpose()
    # Define the colors
    cmap = ListedColormap(['#fde27dff', '#ec7063ff', '#5dade2ff'])
    ax_phase.pcolormesh(kappa, stim, phase_plot, cmap=cmap, vmin=0, vmax=2, rasterized=True)
    # set limits
    ax_phase.set_xlim([0, kappa_max])
    ax_phase.set_ylim([0, 1])
    # set ticks
    # ax_phase.set_yticks([0, 1])
    # ax_phase.set_xticks([0, kappa_max])
    # labels
    ax_phase.set_xlabel("stimulation heterogeneity $\\kappa$")
    ax_phase.set_ylabel("intrinsic stimulation $S$")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase transtion between decision for target 1 and target 2
def plot_phases_ind_s_kappa(env_prop, beh_prop, ang_diff, s, kappa, fig_name, seed=None):
    """
    Plot decision-making between two targets for a fixed angular difference between the targets, fixed width and strength
    of the first cue, and variable width and strength of the second cue.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param ang_diff: angular difference between the targets
    :type ang_diff: float
    :param s: strength of the cue
    :type s: float
    :param kappa: heterogeneity of the cue
    :type kappa: float
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig,ax = plt.subplots(figsize=(2.5, 2.5), layout="constrained")
    # create axes
    num = 25
    rep = 10
    strengths = np.linspace(0,1,num+1)
    widths = np.linspace(0,np.pi,num+1)
    phase_plot = np.zeros((num,num))
    # hold the animal in a fixed position for the duration of the experiment, fix initial heading
    beh_prop.init_hold = env_prop.time_max
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st0 in range(num):
        # find the kappa corresponding to the given width
        def equation(kapp):
            return special.i0(kapp) * 2 * np.pi * np.exp(-kapp) - widths[st0]
        kappa_var = optimize.fsolve(equation, 1.0)[0]
        for st1 in range(num):
            print("Generated point: " + str(st0) + "/" + str(num) + " and " + str(st1) + "/" + str(num))
            # find the strength
            s_var = strengths[st1]
            for rp in range(rep):
                # create pattern
                beh_prop_var = copy.deepcopy(beh_prop)
                beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2*beh_prop_var.brain_fp, np.pi / 4, 0]
                beh_prop_var.cue = [[2, False, kappa,s, -ang_diff/2],[2, False, kappa_var,s_var, ang_diff/2]]
                ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
                pattern = Pattern(env_prop,ind_prop)
                # simulate the pattern
                pattern.simulate(True)
                # obtain the intended heading direction
                angles = np.linspace(0, 2 * np.pi-env_prop.angle_step, env_prop.angle_num)
                heading_y = np.sum(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num]*np.sin(angles)) # intended heading - y component
                # heading_x = np.sum(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num]*np.cos(angles)) # intended heading - x component
                # angle_heading = np.arctan2(heading_y,heading_x)
                if heading_y > 0:
                    phase_plot[st0][st1] += 1
            phase_plot[st0][st1] /= rep
    # make the plot
    phase_plot = phase_plot.transpose()
    ax.pcolormesh(widths, strengths, phase_plot, cmap="Greys", vmin=0, vmax=1, rasterized=True)
    ax.scatter([special.i0(kappa) * 2 * np.pi * np.exp(-kappa)],[s],color='#c83737ff')
    # set limits
    ax.set_xlim(0,np.pi)
    ax.set_ylim(0, 1)
    # set ticks
    ax.set_xticks([0,np.pi/2,np.pi])
    ax.set_yticks([0, 0.5,1])
    ax.set_xticklabels([0, "$\\pi/2$", "$\\pi$"])
    # labels
    ax.set_ylabel("strength $s_0$")
    ax.set_xlabel("width $2 \\pi I_0(\\kappa)e^{-\\kappa}$")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase transtion between decision for target 1 and target 2
def plot_phases_ind_kappas(env_prop, beh_prop, ang_diff, s, kappa_max, fig_name, seed=None):
    """
    Plot decision-making between two targets for a fixed angular difference between the targets, strength
    of the cues, and variable width of the cues.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param ang_diff: angular difference between the targets
    :type ang_diff: float
    :param s: strength of the cues
    :type s: list[float]
    :param kappa_max: maximal heterogeneity of the cues
    :type kappa_max: float
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig,ax = plt.subplots(figsize=(2.5, 2.5), layout="constrained")
    # create axes
    num = 25
    rep = 10
    kappas = np.linspace(0,kappa_max,num+1)
    phase_plot = np.zeros((num,num))
    # hold the animal in a fixed position for the duration of the experiment, fix initial heading
    beh_prop.init_hold = env_prop.time_max
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st0 in range(num):
        # find the kappa corresponding to cue 0
        kappa0 = (kappas[st0]+kappas[st0+1])/2
        for st1 in range(num):
            print("Generated point: " + str(st0) + "/" + str(num) + " and " + str(st1) + "/" + str(num))
            # find the kappa corresponding to cue 1
            kappa1 = (kappas[st1]+kappas[st1+1])/2
            for rp in range(rep):
                # create pattern
                beh_prop_var = copy.deepcopy(beh_prop)
                beh_prop_var.init_brain = [3, beh_prop_var.brain_fp, 2*beh_prop_var.brain_fp, np.pi / 4, 0]
                beh_prop_var.cue = [[2, False, kappa0,s[0], -ang_diff/2],[2, False, kappa1,s[1], ang_diff/2]]
                ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
                pattern = Pattern(env_prop,ind_prop)
                # simulate the pattern
                pattern.simulate(True)
                # obtain the intended heading direction
                angles = np.linspace(0, 2 * np.pi-env_prop.angle_step, env_prop.angle_num)
                heading_y = np.sum(pattern.nx[pattern.pt_now, 0, :pattern.env_prop.angle_num]*np.sin(angles)) # intended heading - y component
                if heading_y > 0:
                    phase_plot[st0][st1] += 1
            phase_plot[st0][st1] /= rep
    # make the plot
    phase_plot = phase_plot.transpose()
    ax.pcolormesh(kappas, kappas, phase_plot, cmap="Greys", vmin=0, vmax=1, rasterized=True)
    # set limits
    ax.set_xlim(0,kappa_max)
    ax.set_ylim(0, kappa_max)
    # labels
    ax.set_xlabel("cue 1 heterogeneity $\\kappa_1$")
    ax.set_ylabel("cue 2 heterogeneity $\\kappa_2$")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create a plot of the phase transtion between decision for target 1 and target 2
def plot_phases_ind_distance(env_prop, beh_prop, width, s, dist_max, fig_name, seed=None):
    """
    Plot decision-making between two targets for a fixed angular difference between the targets as the distance to a target varies.

    :param env_prop: Properties describing the environmental context.
    :type env_prop: EnvProp
    :param beh_prop: Attributes and parameters used to initialize the behavioral properties of the individuals BehProp.
    :type beh_prop: Union(None,dict)
    :param width: width of the cues (i.e. their radius)
    :type width: float
    :param s: strength of the cues
    :type s: list[float]
    :param dist_max: distance between the targets
    :type dist_max: float
    :param fig_name: The name of the file to store the resulting heatmap plots.
    :type fig_name: str
    :param seed: (Optional) The seed value for random number generation to reproduce results.
    :type seed: Union(int,None)
    """
    # seed the random generator
    if seed is not None:
        np.random.seed(seed)
    # prepare plot
    fig,ax = plt.subplots(figsize=(2.5, 2.5/2*1.1), layout="constrained")
    # create axes
    num = 50
    rep = 50
    distance = np.linspace(0,dist_max,num)
    phase_plot = np.zeros(num)
    # for each value of stim, create pattern and save stim_plot and phase_plot
    for st0 in range(num):
        d = distance[st0]
        print("Generated point: " + str(st0) + "/" + str(num))
        for rp in range(rep):
            # create pattern
            beh_prop_var = copy.deepcopy(beh_prop)
            beh_prop_var.init_brain = [0]
            beh_prop_var.cue = [[3, False, width,s[0], 0,-width],[3, False, width,s[1], 0,dist_max+width]]
            beh_prop_var.init_pos = (0,d)
            ind_prop = [IndProp([beh_prop_var], np.asarray([[0]]))]
            pattern = Pattern(env_prop,ind_prop)
            # simulate the pattern
            while pattern.time <= pattern.env_prop.time_max and np.abs(pattern.nx[pattern.pt_now, 0, pattern.env_prop.angle_num+2]) < np.pi/4:
                pattern.update_pattern(True)
            # obtain the intended heading direction
            if pattern.nx[pattern.pt_now, 0, pattern.env_prop.angle_num+2]<0:
                phase_plot[st0] += 1
        phase_plot[st0] /= rep
    # make the plot
    phase_plot = phase_plot.transpose()
    ax.plot(distance,phase_plot,color='black')
    ax.fill_between(distance, -0.02, phase_plot, color='lightgray', alpha=0.5)
    # set limits
    ax.set_xlim(0,dist_max)
    ax.set_ylim(-0.02, 1.1)
    # labels
    ax.set_xlabel("distance to cue 1")
    ax.set_ylabel("probability to reach cue 1 first")
    # save the plot
    fig.savefig("Figures/"+fig_name+".svg")
    fig.savefig("Figures/"+fig_name+".png", dpi=300, format='png')

# Create trajectories in the background for the
# SEMI-ANALYTICAL TECHNIQUES FOR BIFURCATIONS BETWEEN TWO TARGETS
class TwoTargets:
    def __init__(self,interaction,T,strength,S,phi):
        # parameters of neural field model
        self.int = interaction          # self-interaction
        self.T = T                      # neural noise
        self.S = S                      # intrinsic stimulation
        self.strength = strength        # strength of stimulus
        self.phi = phi

    def f0(self,n):
        J_self = self.int[1]
        J_diff = self.int[1] * np.cos(np.pi * (np.abs(self.phi) / np.pi) ** self.int[2])
        interaction = (J_self * n[0] + J_diff * n[1]) / self.T
        stimulation = np.log(self.strength[0]*self.S)
        return -n[0]+1/(1+np.exp(-interaction-stimulation))

    def f1(self,n):
        J_self = self.int[1]
        J_diff = self.int[1] * np.cos(np.pi * (np.abs(self.phi) / np.pi) ** self.int[2])
        interaction = (J_self * n[1] + J_diff * n[0]) / self.T
        stimulation = np.log(self.strength[1]*self.S)
        return -n[1]+1/(1+np.exp(-interaction-stimulation))

    def forcing(self,n,t):
        return [self.f0(n), self.f1(n)]

    def solve(self,n_init,t_max,t_step):
        t = np.arange(0,t_max+t_step,t_step)
        sol = integrate.odeint(self.forcing, n_init, t)
        return sol

    def plot_phase_portrait(self,ax):
        N1, N0 = np.mgrid[0:1:100j, 0:1:100j]
        f0 = self.f0([N0, N1])
        f1 = self.f1([N0, N1])
        ax.streamplot(N0, N1, f0, f1, density=[0.8,1], color="black", linewidth=0.5)
        ax.set_xlim([0,1])
        ax.set_ylim([0, 1])
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xlabel("$n_1$")
        ax.set_ylabel("$n_2$")

# determines angle at which decision for target 1 is stable for strong initial bias
def find_binoidal1(interaction, T, strength, S, error, t_max):
    theta_min = 0
    theta_max = np.pi
    theta = (theta_max + theta_min) / 2
    targets = TwoTargets(interaction, T,strength, S, theta)
    while theta_max-theta_min>error:
        theta = (theta_max+theta_min)/2
        targets.phi = theta
        sol=targets.solve([1,0],t_max,1e-1)
        if sol[-1][0]<0.5 or sol[-1][1]>0.5:
            theta_min = theta
        else:
            theta_max = theta
    return (theta_max+theta_min)/2

# determines angle at which decision for target 2 is stable for strong initial bias
def find_binoidal2(interaction, T, strength, S, error, t_max):
    theta_min = 0
    theta_max = np.pi
    theta = (theta_max + theta_min) / 2
    targets = TwoTargets(interaction, T,strength, S, theta)
    while theta_max - theta_min > error:
        theta = (theta_max + theta_min) / 2
        targets.phi = theta
        sol=targets.solve([0,1],t_max,1e-2)
        if sol[-1][0]>0.5 or sol[-1][1]<0.5:
            theta_min = theta
        else:
            theta_max = theta
    return (theta_max+theta_min)/2

# determines angle at which compromise direction bifurcates with decision direction
def find_spinoidal(interaction, T, strength, S, error, t_max):
    theta_min = 0
    theta_max = np.pi
    theta = (theta_max + theta_min) / 2
    targets = TwoTargets(interaction, T,strength, S, theta)
    while theta_max - theta_min > error:
        theta = (theta_max + theta_min) / 2
        targets.phi = theta
        #sol11 = targets.solve([1, 1 - 1e-9], t_max, 1e-1)
        #sol10 = targets.solve([1, 0], t_max, 1e-1)
        sol=targets.solve([1,1-1e-9],t_max,1e-1)
        if sol[-1][0] > 0.5 and sol[-1][1] > 0.5:
        #if sol11[-1][1]>0.5 or np.linalg.norm(sol11[-1]-sol10[-1])>1e-1:
            theta_min = theta
        else:
            theta_max = theta
    return (theta_max+theta_min)/2

# plots phase diagram of spinoidal and binoidal bifurcations
def plot_phase_diagram(interaction,T,s1,s2_max, S, fig_name):
    # prepare plotted curves
    s_num = 500
    strength = np.linspace(s1,s2_max,s_num)
    binoidal1= np.zeros(s_num)
    binoidal2 = np.zeros(s_num)
    spinoidal = np.zeros(s_num)
    # find spinoidal and binoidals
    for s in range(s_num):
        binoidal1[s] = find_binoidal1(interaction, T, [1-s1,1-strength[s]], S, 1e-4, 500)
        binoidal2[s] = find_binoidal2(interaction, T, [1-s1,1-strength[s]], S, 1e-4, 500)
        spinoidal[s] = find_spinoidal(interaction, T, [1-s1,1-strength[s]], S, 1e-4, 500)
    # prepare figure
    fig = plt.figure(figsize=(5, 5), layout="constrained")
    # crate axes
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    # make plot
    ax.plot(binoidal1,strength-s1, color="gold", label="binoidal 1")
    ax.plot(binoidal2,strength-s1, color="dimgray", label="binoidal 2")
    ax.plot(spinoidal,strength-s1, color="black",label="spinoidal")
    # set ticks
    #ax.set_yticks([0,0.05,0.1,0.15])
    #ax.set_yticklabels([np.round(k*(1-strength_min)/5,2) for k in range(6)])
    #ax.set_xticks([k * np.pi / 12 for k in range(13)])
    #ax.set_xticklabels(["0", "$\\pi$/12", "$\\pi$/6","$\\pi$/4", "$\\pi$/3","$5\\pi$/12", "$\\pi$/2", "$7\\pi$/12", "$2\\pi$/3", "$3\\pi$/4","$5\\pi$/6","$11\\pi$/12","$\\pi$"])
    # set limits
    x_min = min(binoidal1.min(), binoidal2.min(), spinoidal.min())
    x_max = max(binoidal1.max(), binoidal2.max(), spinoidal.max())
    x_mid = (x_min + x_max) / 2
    x_dist = x_max - x_min
    x_min = max(x_mid - 3*x_dist/4, 0)
    x_max = min(x_mid + 3*x_dist/4, np.pi)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, s2_max-s1])
    # set labels
    ax.set_xlabel("angle between targets $\\phi$")
    ax.set_ylabel("stimulation difference $\\Delta s$")
    ax.legend()
    # save the figure
    fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
    fig.savefig("Figures//" + fig_name + ".svg")

# determines spinoidal line in x,y coordinates for a fixed value y, crit_angle and targets at x=pos_max[0] and y=0,pos_max[1]
def find_crit_x(pos_max, crit_angle, y):
    if crit_angle>np.pi/2:
        return pos_max[0]+(-pos_max[1]+np.sqrt(pos_max[1]**2-4*(y-pos_max[1])*y*np.tan(crit_angle)**2))/(2*np.tan(crit_angle))
    else:
        return pos_max[0]+(-pos_max[1]-np.sqrt(pos_max[1]**2-4*(y-pos_max[1])*y*np.tan(crit_angle)**2))/(2*np.tan(crit_angle))