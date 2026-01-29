import math
import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, goals):
        """
        goals can be a list of tuples (angle, radius, strength, kappa)
        or dicts with optional 'start_step'/'end_step' for scheduled goals.
        You can also include 'kappa_schedule' with fields {start, end, duration, scale}
        to ramp the goal concentration over time. scale can be 'linear' (default) or 'log'.
        Additionally, kappa_schedule can define pulses with keys
        {pulse_every, pulse_duration, pulse_value, pulse_offset, pulse_growth}
        to periodically boost kappa. pulse_growth increases the interval after each pulse.
        """
        self.goal_specs = [self._normalize_goal(g) for g in goals]
        # Track the current active (x, y) goals for logging/compatibility
        self.goals = self.get_goal_positions(step_idx=0)

    def _normalize_goal(self, goal):
        if isinstance(goal, dict):
            spec = goal.copy()
            required = ["angle", "radius", "strength", "kappa"]
            for key in required:
                if key not in spec:
                    raise ValueError(f"Goal spec missing required key '{key}'")
        else:
            angle, radius, strength, kappa = goal
            spec = {"angle": angle, "radius": radius, "strength": strength, "kappa": kappa}
        spec.setdefault("start_step", 0)
        spec.setdefault("end_step", None)  # None == active indefinitely
        return spec

    @staticmethod
    def polar_to_cartesian(angle, radius):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return (x, y)

    def _active_goal_specs(self, step_idx=None):
        """Filter goals based on start/end schedule."""
        if step_idx is None:
            return list(self.goal_specs)
        active = []
        for spec in self.goal_specs:
            start = spec.get("start_step", 0) or 0
            end = spec.get("end_step", None)
            if step_idx < start:
                continue
            if end is not None and step_idx >= end:
                continue
            active.append(spec)
        return active

    def get_goal_positions(self, step_idx=None):
        active_specs = self._active_goal_specs(step_idx)
        positions = [self.polar_to_cartesian(spec["angle"], spec["radius"]) for spec in active_specs]
        self.goals = positions  # keep legacy attribute in sync
        return positions

    def get_goal_info(self, agent_position, step_idx=None):
        """
        Return a list of (angle_to_goal, strength, uncertainty) for active goals.
        """
        goal_info = []
        for spec in self._active_goal_specs(step_idx):
            goal = self.polar_to_cartesian(spec["angle"], spec["radius"])
            vector_to_goal = np.array(goal) - agent_position
            angle_to_goal = np.arctan2(vector_to_goal[1], vector_to_goal[0])
            distance = np.linalg.norm(vector_to_goal)
            strength = self._resolve_strength(spec, distance)
            kappa = self._resolve_kappa(spec, step_idx)
            goal_info.append((angle_to_goal, strength, kappa))
        return goal_info

    def update_goals(self, delta_angle=0.003):
        """
        Rotate each goal around the origin and regenerate its (x, y).
        """
        self.goal_specs = [
            {
                **spec,
                "angle": (spec["angle"] + delta_angle) % (2 * np.pi),
            }
            for spec in self.goal_specs
        ]

    def goal_reached(self, agent_position, step_idx=None):
        """
        Check if the agent_position is within 0.5 units of any active goal.
        """
        for index, goal in enumerate(self.get_goal_positions(step_idx)):
            distance = np.linalg.norm(np.array(goal) - np.array(agent_position))
            if distance < 0.5:
                return index  # Return the index of the goal
        return None       # Return None if no goal is reached

    def _resolve_kappa(self, spec, step_idx):
        """
        Optionally ramp kappa from start->end over a duration once the goal is active.
        """
        kappa = spec["kappa"]
        if step_idx is None:
            return kappa

        schedule = spec.get("kappa_schedule")
        if not schedule:
            return kappa

        # Pulses override ramps when active
        pulse_every = schedule.get("pulse_every")
        if pulse_every:
            pulse_every = int(pulse_every)
            pulse_duration = max(1, int(schedule.get("pulse_duration", 1)))
            pulse_value = float(schedule.get("pulse_value", kappa))
            pulse_offset = int(schedule.get("pulse_offset", 0))
            pulse_growth = int(schedule.get("pulse_growth", 0))  # additive increase per interval
            start_step = spec.get("start_step", 0) or 0
            rel_t = step_idx - start_step - pulse_offset
            if rel_t >= 0:
                # Walk intervals until we pass rel_t (safe for modest step counts)
                interval = max(1, pulse_every)
                elapsed = 0
                max_iter = 10000  # guard against runaway loops
                for _ in range(max_iter):
                    if rel_t < elapsed:
                        break
                    if elapsed <= rel_t < elapsed + pulse_duration:
                        return pulse_value
                    elapsed += interval
                    interval = max(1, interval + pulse_growth)

        start_kappa = schedule.get("start", kappa)
        end_kappa = schedule.get("end", start_kappa)
        duration = max(1, int(schedule.get("duration", 1)))
        scale = schedule.get("scale", "linear").lower()
        mode = schedule.get("mode", "").lower()
        use_cosine = bool(schedule.get("cosine", False) or mode == "cosine" or "cosine_min" in schedule or "cosine_max" in schedule)

        # Align ramp to the goal's start_step so it begins when the goal is active.
        start_step = spec.get("start_step", 0) or 0
        t = max(0, step_idx - start_step)
        frac = min(1.0, t / duration)

        if use_cosine:
            cos_min = float(schedule.get("cosine_min", start_kappa))
            cos_max = float(schedule.get("cosine_max", end_kappa if end_kappa is not None else start_kappa))
            period = float(schedule.get("cosine_period", schedule.get("period", duration)))
            period_growth = float(schedule.get("cosine_period_growth", 0.0))
            current_period = max(1e-6, period + period_growth * t)
            mid = 0.5 * (cos_min + cos_max)
            amp = 0.5 * (cos_max - cos_min)
            phase = 2.0 * math.pi * (t / current_period)
            return float(mid + amp * math.cos(phase))

        if scale == "log":
            # Log-interpolate for smoother decay when small kappas matter most
            if start_kappa <= 0 or end_kappa <= 0:
                return start_kappa + (end_kappa - start_kappa) * frac
            start_log = np.log(start_kappa)
            end_log = np.log(end_kappa)
            return float(np.exp(start_log + (end_log - start_log) * frac))

        return start_kappa + (end_kappa - start_kappa) * frac

    def _resolve_strength(self, spec, distance):
        """
        Optionally adjust strength based on distance to the goal.
        Supported schedule: distance_strength={"mode":"linear","half_distance":100,"min_scale":0.0,"max_scale":1.0}
        where scale=1 at distance=0, scale=0.5 at distance=half_distance, scale=0 at distance=2*half_distance.
        """
        strength = float(spec.get("strength", 0.0))
        sched = spec.get("distance_strength")
        if not sched:
            return strength

        mode = str(sched.get("mode", "linear")).lower()

        min_scale = float(sched.get("min_scale", 0.0))
        max_scale = float(sched.get("max_scale", 1.0))

        if mode == "linear":
            half_distance = float(sched.get("half_distance", 100.0))
            if half_distance <= 0:
                return strength
            scale = 1.0 - (distance / (2.0 * half_distance))
            scale = max(min_scale, min(max_scale, scale))
            return strength * scale

        if mode == "log":
            target_distance = float(sched.get("target_distance", sched.get("half_distance", 15.0)))
            target_scale = float(sched.get("target_scale", 1e-4))
            target_scale = max(target_scale, 1e-12)
            # Solve scale = max_scale * exp(-d / lambda), lambda = -d / ln(target_scale/max_scale)
            if max_scale <= 0 or target_distance <= 0:
                return strength
            lam = -target_distance / max(np.log(target_scale / max_scale), -1e6)
            if lam <= 1e-9:
                return strength
            scale = max_scale * math.exp(-distance / lam)
            scale = max(min_scale, min(max_scale, scale))
            return strength * scale

        return strength
