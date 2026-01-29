import numpy as np

class PerceptionSystem:
    def __init__(self, num_groups, num_spins_per_group, suppression_scaling, log_transform=True, normalize_kernel=False):
        """
        Initialize the perception system.

        :param num_groups: Number of direction groups.
        :param num_spins_per_group: Number of spins per group.
        :param suppression_scaling: Global suppression factor S (used in log).
        :param log_transform: Whether to apply log to the accumulated field.
        :param normalize_kernel: If True, normalize the *output* field (log if enabled)
                                 so comparisons are shape-focused, not scale-focused.
        """
        self.num_groups = num_groups
        self.num_spins_per_group = num_spins_per_group
        self.suppression_scaling = suppression_scaling  # This is S
        self.log_transform = log_transform
        self.normalize_kernel = normalize_kernel
        self.group_angles = np.linspace(0, 2 * np.pi, num_groups, endpoint=False)

    def perceive(self, stimuli):
        """
        Compute log-suppression-based perceptual field h_i for each spin.

        :param stimuli: List of tuples (angle, strength, kappa)
                        where angle = θ_g, strength = S_g, kappa = κ_g
        :return: Array of log-perceptual input p_i for each spin.
        """
        eps = 1e-10  # Numerical safety to avoid log(0)
        p_per_group = np.zeros(self.num_groups)

        for (angle_g, strength_g, kappa_g) in stimuli:
            if kappa_g <= 0:
                kappa_g = 1e-6  # Avoid numerical issues

            angle_diffs = self.group_angles - angle_g
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

            kernel = np.exp(kappa_g * (np.cos(angle_diffs) - 1))  # shifted von Mises kernel
            weighted = strength_g * kernel  # element-wise

            p_per_group += weighted  # accumulate over all stimuli

        # Multiply by global suppression scaling S
        p_per_group *= self.suppression_scaling

        if self.log_transform:
            # Logarithmic transformation (safeguard small values)
            p_per_group = np.log(np.maximum(p_per_group, eps))
        else:
            p_per_group = np.maximum(p_per_group, 0.0)

        if self.normalize_kernel:
            mean_val = np.mean(p_per_group)
            if np.isfinite(mean_val):
                if self.log_transform:
                    p_per_group = p_per_group - mean_val  # center log field
                else:
                    if abs(mean_val) > eps:
                        p_per_group = p_per_group / mean_val
        else:
            # No normalization; keep as-is
            pass

        # Expand to spins
        perception = np.repeat(p_per_group, self.num_spins_per_group)

        return perception
