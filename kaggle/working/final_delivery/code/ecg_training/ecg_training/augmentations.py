import numpy as np


class ECGAugmenter:
    def __init__(self, config):
        self.enabled = bool(config.get("enabled", False))
        self.noise_std = float(config.get("noise_std", 0.0))
        self.baseline_wander_std = float(config.get("baseline_wander_std", 0.0))
        lo, hi = config.get("amplitude_scale_range", [1.0, 1.0])
        self.scale_range = (float(lo), float(hi))
        self.time_shift_samples = int(config.get("time_shift_samples", 0))
        self.lead_dropout_prob = float(config.get("lead_dropout_prob", 0.0))
        self.time_mask_prob = float(config.get("time_mask_prob", 0.0))
        self.time_mask_max_width = int(config.get("time_mask_max_width", 0))

    def __call__(self, signal):
        if not self.enabled:
            return signal

        augmented = np.array(signal, dtype=np.float32, copy=True)
        leads, length = augmented.shape

        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        augmented *= scale

        if self.noise_std > 0.0:
            augmented += np.random.normal(0.0, self.noise_std, size=augmented.shape).astype(np.float32)

        if self.baseline_wander_std > 0.0:
            time_axis = np.linspace(0.0, 2.0 * np.pi, num=length, dtype=np.float32)
            phase = np.random.uniform(0.0, 2.0 * np.pi)
            baseline = self.baseline_wander_std * np.sin(time_axis + phase)
            augmented += baseline[None, :]

        if self.time_shift_samples > 0:
            shift = np.random.randint(-self.time_shift_samples, self.time_shift_samples + 1)
            if shift != 0:
                augmented = np.roll(augmented, shift=shift, axis=1)

        if self.lead_dropout_prob > 0.0:
            for lead_idx in range(leads):
                if np.random.rand() < self.lead_dropout_prob:
                    augmented[lead_idx] = 0.0

        if self.time_mask_prob > 0.0 and self.time_mask_max_width > 0 and np.random.rand() < self.time_mask_prob:
            width = np.random.randint(1, self.time_mask_max_width + 1)
            start = np.random.randint(0, max(1, length - width))
            augmented[:, start : start + width] = 0.0

        return augmented
