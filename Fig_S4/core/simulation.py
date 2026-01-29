import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.ticker import MultipleLocator
from .spinSystem import SpinSystem
from .perceptionSystem import PerceptionSystem
from .environment import Environment
from .agent import Agent

class Simulation:
    def __init__(self, goals, num_groups, num_spins_per_group, suppression_scaling, intrinsic_stimulation, T, J, nu, step_size, initial_position, initial_orientation,
                 stop_on_goal=False, time_delay=0, moving_goal=False,
                 p_spin_up=0.5, velocity_scaling='constant',
                 egocentric=False, prerun=False, prerun_config=None, dynamics = 'glauber', dt= 1, tau = 160,
                 log_perception_input=True, normalize_perception_kernel=False,
                 resense_static_each_step=False):

        print(f"Initializing Simulation: time_delay={time_delay} (type: {type(time_delay)}), "
              f"suppression_scaling={suppression_scaling} (type: {type(suppression_scaling)})")

        # Basic components of the simulation
        self.environment = Environment(goals)
        self.perception_system = PerceptionSystem(
            num_groups=num_groups,
            num_spins_per_group=num_spins_per_group,
            suppression_scaling=suppression_scaling,
            log_transform=log_perception_input,
            normalize_kernel=normalize_perception_kernel,
        )
        self.spin_system = SpinSystem(
            num_groups=num_groups,
            num_spins_per_group=num_spins_per_group,
            T=T, J=J, nu=nu,
            p_spin_up=p_spin_up,
            time_delay=time_delay,
            intrinsic_stimulation=intrinsic_stimulation,
            dynamics = dynamics,
            dt = dt,
            tau = tau
        )
        self.agent = Agent(
            spin_system=self.spin_system,
            perception_system=self.perception_system,
            environment=self.environment,
            step_size=step_size,
            initial_position=initial_position,
            velocity_scaling=velocity_scaling,
            initial_orientation=initial_orientation
        )

        # Bookkeeping
        self.prerun = prerun
        self.egocentric = egocentric
        self.stop_on_goal = stop_on_goal
        self.moving_goal = moving_goal
        self.resense_static_each_step = resense_static_each_step
        self.reached_goal_index = None
        
        #Default prerun_config
        default_prerun_config = {
            'distribution': "uniform",
            'prerun_steps': 1000,
            'peak_angle': np.pi / 2,
            'rounds': 1.0
        }
        
        # Merge user config with defaults
        self.prerun_config = {**default_prerun_config, **(prerun_config or {})}

        # Data collection
        self.heading = []
        self.agent_positions = []
        self.spin_states = []
        self.goals = []
        self.width = []
        self.direction = []
        self.perception_inputs = []  # mean external field per group over time

    def perform_prerun_setup(self, distribution="uniform", prerun_steps=1000,
                             peak_angle=np.pi/2, rounds=1.0):
        """
        Runs an initial pre-run phase to let the spin system equilibrate
        or measure p_spin_up. Afterwards, re-initializes spins according to
        `distribution` (e.g. uniform, von Mises, moving_bump).

        Parameters
        ----------
        distribution : str
            Which distribution to use. Options: 'uniform', 'von_mises', 'moving_bump', ...
        prerun_steps : int
            Number of spin-system steps for the pre-run.
        peak_angle : float
            Used by distributions that require an angular “center” (e.g. von Mises, bump).
        rounds : float
            For ‘moving_bump’, how many full rotations (2π each) the bump should
            traverse over the spin history of length = `time_delay`.
        """

        print("Prerun is true, starting prerun setup.")
        # 1) Record initial states and let the agent sense
        self.agent_positions.append(self.agent.get_position())
        self.heading.append(self.agent.orientation)
        self.spin_states.append(self.spin_system.get_states())
        self.goals.append(self.environment.goals)
        self.width.append(self.spin_system.get_width_of_activity())
        self.direction.append(self.spin_system.average_direction_of_activity())
        self.agent.sense()
        self._record_perception_input()

        # 2) Run spin system for some steps to equilibrate
        for _ in range(prerun_steps):
            self.spin_system.step(timedelay=False)

        # 3) Compute final average spin; store it in spin_system
        self.spin_system.p_spin_up = np.mean(self.spin_system.spins)
        print("Prerun ended with p_spin_up =", self.spin_system.p_spin_up)

        # 4) Clear spin history and initialize fresh distribution
        self.spin_system.spins_history.clear()
        self.initialize_spin_distribution(
            distribution=distribution,
            peak_angle=peak_angle,
            rounds=rounds
        )

    def initialize_spin_distribution(self, distribution="uniform",
                                     peak_angle=np.pi/2, rounds=1):
        """
        Central method that (re)initializes the spin configuration
        according to `distribution`, and populates spins_history.

        Parameters
        ----------
        distribution : str
            'uniform', 'von_mises', 'moving_bump', ...
        peak_angle : float
            Angle in radians that can serve as the center of a distribution.
        rounds : float
            For ‘moving_bump’, how many 2π rotations we want over time_delay steps.
        """

        # Create a dictionary of distribution initializers
        init_methods = {
            "uniform":     self._init_uniform,
            "von_mises":   self._init_von_mises,
            "moving_bump": self._init_moving_bump,
            "zigzag_bump":   self._init_zigzag_bump
        }

        if distribution not in init_methods:
            raise ValueError(f"Unknown distribution type: {distribution}")

        # Call the appropriate initialization method
        init_func = init_methods[distribution]
        init_func(peak_angle=peak_angle, rounds=rounds)

    def _init_uniform(self, peak_angle=0.0, rounds=1.0, kappa=2.0):
        """
        Reinitialize spins with a uniform distribution, based on p_spin_up
        determined during the prerun.
        """
        # Create a single spin configuration
        new_spins = np.random.choice(
            [0, 1],
            size=(self.spin_system.num_groups, self.spin_system.num_spins_per_group),
            p=[1 - self.spin_system.p_spin_up, self.spin_system.p_spin_up]
        )

        # Add it to the history once (the left end is “current”)
        self.spin_system.spins_history.appendleft(new_spins)
        # Also set the current spins
        self.spin_system.spins = new_spins.copy()

    def _init_von_mises(self, peak_angle=np.pi/2, rounds=1.0, kappa=2.0):
        """
        Each spin group i has an angle in spin_system.angles.
        We assign p_up_i based on a von Mises “raw” function, then scale so
        that the average p_up_i = self.spin_system.p_spin_up.
        """
        new_spins = self._sample_von_mises(peak_angle=peak_angle, kappa=kappa)
        self.spin_system.spins_history.appendleft(new_spins)
        self.spin_system.spins = new_spins.copy()

    def _sample_von_mises(self, peak_angle=np.pi/2, kappa=2.0):
        """
        Return an (num_groups x num_spins_per_group) array of 0/1 spins 
        drawn from a von Mises distribution centered at `peak_angle` with 
        concentration `kappa`, scaled so the average spin-up == p_spin_up.
        """
        # All angles in one 1D array
        angles = self.spin_system.angles  # length = num_groups * num_spins_per_group
        N = angles.size
        p_target = self.spin_system.p_spin_up

        # 1) raw_i = exp(kappa * cos(angle_i - peak_angle))
        raw = np.exp(kappa * np.cos(angles - peak_angle))

        # 2) We'll normalize these raw values so that the average p_i is p_target.
        #    We do: p_i = p_target * [raw_i / average(raw)]
        #    where average(raw) = (1/N)*sum(raw_i).
        avg_raw = np.mean(raw)
        p_up_array = p_target * (raw / avg_raw)  # shape = (N,)

        # 3) Sample from Bernoulli(p_up_array)
        spins_1d = (np.random.rand(N) < p_up_array).astype(int)

        # 4) Reshape to (num_groups, num_spins_per_group)
        num_groups = self.spin_system.num_groups
        num_spins_per_group = self.spin_system.num_spins_per_group
        new_spins = spins_1d.reshape((num_groups, num_spins_per_group))
        return new_spins

    def _init_moving_bump(self, peak_angle=0.0, rounds=1.0, kappa=2.0):
        """
        Similar to von Mises, except we fill the entire spins_history for
        time_delay+1 steps. The 'peak_angle' is shifted each step so that 
        the bump effectively 'moves' around the circle over time.
        
        We still ensure each step's distribution has an average spin-up 
        of p_spin_up by normalizing similarly to the von Mises approach,
        only here we do it for each step with a slightly incremented angle.
        
        The total rotation for time_delay steps is 2π * rounds.
        """
        td = self.spin_system.time_delay
        if td < 1:
            # If no time_delay, just do one step
            new_spins = self._sample_von_mises(peak_angle=peak_angle, kappa=kappa)
            self.spin_system.spins_history.appendleft(new_spins)
            self.spin_system.spins = new_spins.copy()
            return

        # Angle shift each step
        d_angle = (2.0 * np.pi * rounds) / float(td)

        # We'll build states from oldest to newest
        all_states = []
        for i in range(td + 1):
            current_angle = peak_angle + i * d_angle
            # each distribution is von Mises-based around 'current_angle'
            state = self._sample_von_mises(peak_angle=current_angle, kappa=kappa)
            all_states.append(state)

        # spins_history is a deque that expects index 0 to be the newest
        # So we append them in reverse order
        for st in reversed(all_states):
            self.spin_system.spins_history.appendleft(st)
        # The newest state is the last in all_states
        self.spin_system.spins = all_states[-1].copy()

    def _init_zigzag_bump(self, peak_angle=0.0, rounds=1.0, kappa=2.0):
        """
        Similar to _init_moving_bump, but after half the time_delay,
        the bump reverses direction. That is, the angle increments
        for the first half, then decrements for the second half.

        The total rotation for the full time_delay is 2π*rounds, but
        half is done in the forward direction and half in reverse.
        """
        td = self.spin_system.time_delay
        if td < 1:
            # If no time_delay, just do one step
            new_spins = self._sample_von_mises(peak_angle=peak_angle, kappa=kappa)
            self.spin_system.spins_history.appendleft(new_spins)
            self.spin_system.spins = new_spins.copy()
            return

        # We'll gather all states from oldest to newest in a list, then
        # reverse them into the spins_history (which expects index 0 to be newest).
        all_states = []

        # We'll define half the delay:
        half_td = td // 2  # integer division
        # The remainder if td is odd
        remainder = td - 2 * half_td

        # The total angle we want to cover in the forward phase is (2π*rounds)/2
        # i.e. half of the total rotation. Then we come back in the backward phase.
        total_angle = 2.0 * np.pi * rounds
        forward_angle = total_angle / 2.0

        # Forward increment per step
        d_angle_forward = forward_angle / float(half_td) if half_td else 0
        # We'll start from peak_angle. At step i, angle = peak_angle + i * d_angle_forward
        for i in range(half_td):
            current_angle = peak_angle + i * d_angle_forward
            state = self._sample_von_mises(peak_angle=current_angle, kappa=kappa)
            all_states.append(state)

        # If there's a remainder, we can decide how to handle it
        # e.g. keep going forward for 1 or remainder steps:
        extra_forward_states = remainder // 2  # e.g. if td is odd
        for i in range(extra_forward_states):
            current_angle = peak_angle + half_td * d_angle_forward + i * d_angle_forward
            state = self._sample_von_mises(peak_angle=current_angle, kappa=kappa)
            all_states.append(state)

        # Now define the reversed (backward) motion. Suppose the final forward angle
        # after half_td + extra_forward_states is forward_end_angle:
        forward_end_angle = peak_angle + (half_td + extra_forward_states) * d_angle_forward

        # The total number of steps left for backward movement
        second_phase_steps = td - (half_td + extra_forward_states)
        # The total backward angle we want to retrace:
        # If we want to come back to peak_angle, that is forward_end_angle - peak_angle
        # Or, if we want to do a symmetrical half rotation in reverse, we can keep going 
        # up to forward_angle in negative direction. Let's assume we come back to
        # peak_angle so it "zigzags" back to the start. Then total angle to go back is
        # (forward_end_angle - peak_angle).
        backward_angle_span = forward_end_angle - peak_angle
        if second_phase_steps > 0:
            d_angle_backward = backward_angle_span / float(second_phase_steps)
        else:
            d_angle_backward = 0

        # We go from 0..(second_phase_steps-1). 
        # For step j, the angle = forward_end_angle - j*d_angle_backward
        for j in range(second_phase_steps):
            current_angle = forward_end_angle - j * d_angle_backward
            state = self._sample_von_mises(peak_angle=current_angle, kappa=kappa)
            all_states.append(state)

        # Now we have td states. But we actually need td+1 if we want "td+1" frames?
        # If you want exactly td+1 states in history, you can add one final step at the end.
        # We'll omit that for clarity unless you specifically want it.

        # Put these states in the spins_history so that the last is the newest
        # spins_history[0] is the newest, so we reverse the list
        for st in reversed(all_states):
            self.spin_system.spins_history.appendleft(st)

        # The newest state is the last in all_states
        self.spin_system.spins = all_states[-1].copy()


    def run_simulation(self, num_steps):
        """
        Main simulation. If `prerun` is True, we call perform_prerun_setup(...).
        Then run the agent + spin system for `num_steps`.
        """
        try:
            if self.prerun:
                self.perform_prerun_setup(**self.prerun_config)
            for step_idx in range(num_steps):
                # Move goals first so sensing/logging is aligned with their current positions
                if self.moving_goal and step_idx > 0:
                    self.environment.update_goals()

                current_goals = self.environment.get_goal_positions(step_idx)
                self.agent_positions.append(self.agent.get_position())
                self.heading.append(self.agent.orientation)
                self.spin_states.append(self.spin_system.get_states())
                self.goals.append(current_goals)
                self.width.append(self.spin_system.get_width_of_activity())
                self.direction.append(self.spin_system.average_direction_of_activity())
                self.agent.sense(step_idx=step_idx)
                self._record_perception_input()

                # Spin system evolves for a few steps
                self.spin_system.run_simulation(steps=10)
                if self.egocentric:
                    self.agent.egocentric_move()
                else:
                    self.agent.move()
                goal_index = self.environment.goal_reached(self.agent.get_position(), step_idx=step_idx)

                if self.stop_on_goal and goal_index is not None:
                    self.reached_goal_index = goal_index  # Save the index of the reached goal
                    break
        except TypeError as e:
            print(f"TypeError encountered with time_delay={self.spin_system.time_delay} (type: {type(self.spin_system.time_delay)}): {e}")
            raise

    def run_static_simulation(self, num_steps):
        """
        Run a simulation where the agent's position does not change.
        Only the initial sensing of the environment is conducted and the spin system is evolved.
        """
        if self.prerun:
                self.perform_prerun_setup(**self.prerun_config)

        for step_idx in range(num_steps):
            if self.moving_goal and step_idx > 0:
                # Update goal angles if requested before we sense/log
                self.environment.update_goals()

            current_goals = self.environment.get_goal_positions(step_idx=step_idx)
            self.agent_positions.append(self.agent.get_position())
            self.heading.append(self.agent.orientation)

            # Re-sense whenever goals can change so external field tracks schedule updates
            if step_idx == 0 or self.resense_static_each_step:
                self.agent.sense(step_idx=step_idx)
                self._record_perception_input()

            self.spin_states.append(self.spin_system.get_states())
            self.goals.append(current_goals)
            self.width.append(self.spin_system.get_width_of_activity())
            self.direction.append(self.spin_system.average_direction_of_activity())

            # Run the spin system for a fixed number of steps
            self.spin_system.run_simulation(steps=10)


    def save_results(self, filename):
        results = {
            'agent_positions': self.agent_positions,
            'spin_states': self.spin_states,
            'goals': self.goals,
            'goal_reached': self.reached_goal_index,
            'num_groups': self.spin_system.num_groups,
            'num_spins_per_group': self.spin_system.num_spins_per_group,
            'width': self.width,
            'average_direction': self.direction,
            'heading': self.agent.orientation
        }
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def _record_perception_input(self):
        """
        Store the current external field as mean per group to keep size manageable.
        """
        try:
            field = np.asarray(self.spin_system.external_field, dtype=float)
            G = self.spin_system.num_groups
            S = self.spin_system.num_spins_per_group
            if field.shape == (G * S,):
                field = field.reshape(G, S).mean(axis=1)
            self.perception_inputs.append(np.array(field, dtype=float, copy=True))
        except Exception:
            # Best-effort; skip recording on failure
            pass
    
