import numpy as np
class Agent:
    def __init__(self, spin_system, perception_system, environment, step_size=1.0, initial_position = (0, 0), initial_orientation=0.0 , velocity_scaling='constant'):
        """
        Initialize the agent with references to its brain (spin system), perceptual system, and environment.
        
        :param spin_system: Instance of SpinSystem.
        :param perceptual_system: Instance of PerceptualSystem.
        :param environment: Instance of Environment.
        :param step_size: Step size for agent movement.
        :param initial_position: Initial position of the agent.
        :param initial_orientation: Initial orientation of the agent in radians.
        :param velocity_scaling: Method to scale the velocity ('constant', 'magnitude', 'width').
        """
        self.spin_system = spin_system
        self.perceptual_system = perception_system
        self.environment = environment
        self.step_size = step_size
        self.position = np.array((initial_position), dtype=np.float64)  # Initialize the agent's position at the origin
        self.orientation = initial_orientation  # Orientation in radians
        self.velocity_scaling = velocity_scaling.lower()

    def sense(self, step_idx=None):
        """
        Use the perceptual system to sense the environment based on the current position and update the spin system.
        """
        # Get the current position and perceive the environment accordingly
        stimuli = self.environment.get_goal_info(self.position, step_idx=step_idx)
        perceptual_outputs = self.perceptual_system.perceive(stimuli)
        self.spin_system.update_external_field(perceptual_outputs)

    def egocentric_sense(self, step_idx=None):
        """
        Use the perceptual system to sense the environment based on the current position and orientation, updating the spin system.
        """
        # Get the stimuli relative to the agent's position and orientation
        stimuli = self.environment.get_goal_info(self.position, step_idx=step_idx)

        # Transform stimuli into the agent's egocentric frame
        egocentric_stimuli = self._transform_to_egocentric(stimuli)

        # Use the egocentric stimuli in the perceptual system
        perceptual_outputs = self.perceptual_system.perceive(egocentric_stimuli)

        # Update the spin system with the perceptual outputs
        self.spin_system.update_external_field(perceptual_outputs)
    
    def egocentric_move(self):
        """
        Move the agent according to egocentric neural activity.
        Neural activity must be rotated into allocentric frame before computing direction.
        """
        # Get neural activity vector from spin system (in egocentric frame)
        goal_vector = self.spin_system.get_goal_direction_vector(frame='egocentric', agent_orientation=self.orientation)
        if goal_vector is None:
            return

        # Heading vector
        h = np.array([np.cos(self.orientation), np.sin(self.orientation)])

        # Turning based on cross product
        turning_signal = goal_vector[0] * h[1] - goal_vector[1] * h[0]

        # Update orientation
        omega_0 = np.pi/16  # turning rate
        self.orientation -= omega_0 * turning_signal
        self.orientation %= 2 * np.pi
        
        # Move forward
        v_0 = self.step_size
        self.position += v_0 * h

    def move(self):
        """
        Move the agent according to the ring attractor model using allocentric neural activity.
        Constant speed; turning based on cross product between heading and neural goal direction.
        """
        # Get neural activity vector from spin system
        goal_vector = self.spin_system.get_goal_direction_vector(frame='allocentric')
        if goal_vector is None:
            return

        # Current heading unit vector
        h = np.array([np.cos(self.orientation), np.sin(self.orientation)])

        # Turning rate: cross product of goal vector and heading
        turning_signal = goal_vector[0] * h[1] - goal_vector[1] * h[0]  # 2D cross product: n ∧ h

        # Update orientation
        omega_0 = np.pi/16  # turn rate constant — tweak this as needed
        self.orientation -= omega_0 * turning_signal
        self.orientation %= 2 * np.pi  # keep within [0, 2π)

        # Move forward in new heading
        v_0 = self.step_size
        dx = v_0 * np.cos(self.orientation)
        dy = v_0 * np.sin(self.orientation)
        self.position += np.array([dx, dy])
            
    def get_position(self):
        """Return the agent's current position as (x, y) coordinates."""
        return tuple(self.position)
    
    def goal_reached(self):
        """Return True if the agent has reached the goal, False otherwise."""
        return self.environment.goal_reached(self.position)

    def _transform_to_egocentric(self, stimuli):
        """
        Transform stimuli from the global frame to the agent's egocentric frame.

        :param stimuli: The stimuli in the global frame.
        :return: Stimuli in the agent's egocentric frame.
        """
        transformed = []
        for angle, strength, kappa in stimuli:
            egocentric_angle = (angle - self.orientation) % (2 * np.pi)
            transformed.append((egocentric_angle, strength, kappa))
        return transformed
