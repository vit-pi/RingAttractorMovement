import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm
from collections import deque
import os
from scipy.special import expit

class SpinSystem:
    def __init__(self, num_groups, num_spins_per_group, T, J, nu, p_spin_up=0.5, time_delay=0, dynamics='glauber', intrinsic_stimulation=0.0, dt=0.1, tau=33):
        """ Initialize the SpinSystem with given parameters. """
        self.num_groups = num_groups
        self.num_spins_per_group = num_spins_per_group
        self.T = T  # Temperature
        self.J = J  # Coupling constant
        self.nu = nu  # Angular dependence exponent
        self.p_spin_up = p_spin_up  # Probability of spin being 1 (up)
        self.spins = np.random.choice([0, 1], size=(self.num_groups, self.num_spins_per_group), p=[1 - self.p_spin_up, self.p_spin_up])
        self.time_delay = time_delay
        self.spins_history = deque(maxlen=time_delay + 1)  # Always use deque to manage spin history
        self.spins_history.append(self.spins.copy())  # Start with the initial state in history
        self.dynamics = dynamics # Dynamics type: 'metropolis' or 'glauber'
        # Assigning one unique angle to each group, replicated for each spin
        group_angles = np.linspace(0, 2 * np.pi, num_groups, endpoint=False)
        self.angles = np.repeat(group_angles, self.num_spins_per_group)  # This ensures angles are 1D
        self.external_field = np.zeros(self.num_groups * self.num_spins_per_group)  # Placeholder for external field strengths
        #fill intrinsic stimulation with value of intrinsic stimulation
        self.intrinsic_stimulation = np.full(self.num_groups * self.num_spins_per_group, intrinsic_stimulation)
        self.dt = dt
        self.tau = tau

    def calculate_hamiltonian(self, state=None):
        """ Calculate the Hamiltonian using the provided state or the historical state if none provided. """
        if state is None:
            if not self.spins_history:  # Check if the deque is empty
                raise ValueError("Spin history is unexpectedly empty")
            state_to_use = self.spins_history[-1]  # Always use the oldest state in the history
        else:
            state_to_use = state  # Use the provided state

        J_matrix = self.calculate_j_matrix()
        spin_interaction_matrix = np.outer(state_to_use.flatten(), state_to_use.flatten())
        H_spin_interactions = -(self.J / (self.num_spins_per_group * self.num_groups)) * (J_matrix * spin_interaction_matrix)
        # include diagonal or not?
        H_spin_interactions = np.sum(np.triu(H_spin_interactions, 1))  # Sum only the upper triangle to avoid double counting

        # Adding external field contribution
        external_field_contribution = -self.T*np.sum(self.external_field * state_to_use.flatten())
        # Adding intrinsic stimulation contribution
        intrinsic_stimulation_contribution = - np.sum(self.intrinsic_stimulation * state_to_use.flatten())

        return H_spin_interactions + external_field_contribution + intrinsic_stimulation_contribution 

    def calculate_j_matrix(self):
        """
        Calculate the interaction matrix J based on angular differences using the class attributes.
        """
        # Calculate the absolute difference in angles between each pair of spins
        angle_diff_matrix = np.abs(np.subtract.outer(self.angles, self.angles))

        # Adjust differences to be within [0, pi]
        angle_diff_matrix = np.minimum(angle_diff_matrix, 2 * np.pi - angle_diff_matrix)
        
        # Normalize differences by pi, raise to the power nu, and apply cosine
        J_matrix = np.cos(np.pi *(angle_diff_matrix / np.pi) ** self.nu)
        
        return J_matrix

    def step(self, timedelay=True):
        """ Perform one Metropolis/Glauber step, optionally using a hybrid historical state or the current state. """
        # Select a random spin
        i = np.random.randint(self.num_groups)
        j = np.random.randint(self.num_spins_per_group)

        if timedelay:
            # Create a hybrid state using the latest historical state
            hybrid_state = self.spins_history[-1].copy()
            hybrid_state[i, j] = self.spins[i, j]  # Update the specific spin to its current state
            state_to_use = hybrid_state
        else:
            # Use the current state directly
            state_to_use = self.spins.copy()

        # Calculate current Hamiltonian using the selected state
        current_hamiltonian = self.calculate_hamiltonian(state_to_use)
        
        # Propose flipping the spin
        state_to_use[i, j] ^= 1
        
        # Calculate new Hamiltonian with the flipped spin
        new_hamiltonian = self.calculate_hamiltonian(state_to_use)
        
        # Calculate change in Hamiltonian
        delta_h = new_hamiltonian - current_hamiltonian

            # Determine acceptance based on dynamics
        if self.dynamics == 'metropolis':
            self._metropolis_acceptance(i, j, delta_h)
        elif self.dynamics == 'glauber':
            self._glauber_acceptance(i, j, delta_h)
        else:
            raise ValueError(f"Unknown dynamics type: {self.dynamics}")

        # Update the history with every metropolis step regardless of timedelay
        self.spins_history.appendleft(self.spins.copy())


    def _metropolis_acceptance(self, i, j, delta_h):
        """Metropolis acceptance condition."""
        if delta_h <= 0 or np.random.random() < np.exp(-delta_h / self.T):
            # Accept the spin flip
            self.spins[i, j] ^= 1
    
    def _glauber_acceptance(self, i, j, delta_h):
        """Glauber acceptance condition with numerically stable sigmoid."""
        G = self.num_groups
        N = self.num_spins_per_group

        x = -delta_h / self.T
        acceptance_prob = (G * N * self.dt) / self.tau * expit(x)

        # Original version (less stable):
        # acceptance_prob = (G * N * self.dt) / self.tau * (1 / (1 + np.exp(delta_h / self.T)))

        acceptance_prob = min(acceptance_prob, 1.0)

        if np.random.random() < acceptance_prob:
            self.spins[i, j] ^= 1


    def run_simulation(self, steps):
        for _ in range(steps):
            self.step()
        return self.spins

    def display_spins(self):
        print(self.spins)


    def get_goal_direction_vector(self, frame='allocentric', agent_orientation=0.0):
        """
        Return the average direction of neural activity as a 2D vector.
        
        :param frame: 'allocentric' or 'egocentric'
        :param agent_orientation: current orientation of the agent (needed for egocentric)
        :return: np.array([nx, ny]) or None
        """
        avg_angle = self.average_direction_of_activity()
        if avg_angle is None:
            return None

        if frame == 'egocentric':
            # Convert egocentric angle to allocentric by rotating it by the agent's orientation
            avg_angle = (avg_angle + agent_orientation) % (2 * np.pi)

        return np.array([np.cos(avg_angle), np.sin(avg_angle)])

    def average_direction_of_activity(self):
        # Flatten the spins array and get the angles corresponding to each spin
        flattened_spins = self.spins.flatten()

        # If all spins are active, return None
        if np.all(flattened_spins == 1):
            return None
    
        unit_vectors = np.exp(1j * self.angles)  # Complex numbers representing unit vectors

        # Mask to select only spins with state 1
        active_mask = flattened_spins == 1

        # Calculate the average direction by summing unit vectors with active mask
        sum_vector = np.sum(unit_vectors[active_mask])

        # Normalize to get the unit direction
        if sum_vector != 0:
            avg_direction = np.angle(sum_vector)  # Returns the angle in radians
        else:
            avg_direction = None  # No active spins

        return avg_direction
    

    def get_width_of_activity(self):
        """
        Calculate the width of the peak of activity by measuring the dispersion of active spin directions.
        """
        # Flatten the spins array and get the angles corresponding to each spin
        flattened_spins = self.spins.flatten()
        angles = self.angles.flatten()
        # Mask to select only spins with state 1
        active_mask = flattened_spins == 1
        active_angles = angles[active_mask]

        if len(active_angles) > 1:
            # Compute the mean resultant length R
            unit_vectors = np.exp(1j * active_angles)
            R = np.abs(np.mean(unit_vectors))
            
            # Ensure R is within [0, 1]
            R = min(max(R, 0.0), 1.0)

            # Circular standard deviation can be approximated by sqrt(-2 * ln(R))
            if R > 0:
                circ_std = np.sqrt(-2 * np.log(R))
            else:
                circ_std = np.pi  # Maximum possible dispersion
            return circ_std
        else:
            return None  # Not enough active spins to compute width
    

    def update_external_field(self, perceptual_outputs):
        """ Update external magnetic field strengths based on perceptual system outputs. """
        self.external_field = perceptual_outputs

    def plot_perception(self):

        group_mean_activity = self.external_field.reshape(self.num_groups,self.num_spins_per_group).mean(axis=1)  # Average activity per group
        normalized_activity = (group_mean_activity + 1) / 2  # Normalize between 0 and 1

        # Plot setup
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        cmap = coolwarm  # Colormap from red (-1) to blue (+1)

        # Create the plot with constant bar height
        bars = ax.bar(self.angles[::self.num_spins_per_group], np.ones_like(self.angles[::self.num_spins_per_group]), # Constant height of 1
                      width=2*np.pi/self.num_groups, bottom=1.5, color=cmap(normalized_activity), edgecolor='black')  
    

        # Customizing the plot
        ax.set_yticklabels([])  # Hide radial ticks
        ax.set_xticks(self.angles[::self.num_spins_per_group])
        #ax.set_xticklabels([f'Group {i+1}' for i in range(self.num_groups)])
        ax.grid(False)  # Turn off the grid

        plt.show()

    def get_states(self):
        """Return the current spin states in their original matrix form."""
        return self.spins.copy()
    
    def set_states(self, states):
        """Set the spin states directly from an external source."""
        if states.shape != self.spins.shape:
            raise ValueError(f"Invalid shape for spin states. Expected {self.spins.shape}, but got {states.shape}.")
        self.spins = states.copy()  # Set the spins to the provided states
        self.spins_history.appendleft(self.spins.copy())  # Update the history as well
