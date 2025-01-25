
import json
import numpy as np
import matplotlib.pyplot as plt


class SignalGenerator:
    def __init__(self, num_points=1000, num_samples=10, noise_std=0.2):
        self.num_points = num_points
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.changes_positions = []

    def generate_trend(self, x):
        num_components = np.random.randint(5, 10)
        trend = np.zeros_like(x)
        for _ in range(num_components):
            trend += self.random_trend_component(x)
        return trend

    def random_trend_component(self, x):
        component_type = np.random.choice(['sin', '+exp', '-exp', 'log', 'step'], p=[0.1, 0.1, 0.1, 0.2, 0.5])
        if component_type == 'sin':
            return np.sin(x / np.random.randint(80, 120)) * np.random.uniform(5, 20)
        elif component_type == '+exp':
            return np.exp(x / np.random.randint(500, 600)) * np.random.uniform(1, 5)
        elif component_type == '-exp':
            return np.exp(-x / np.random.randint(500, 600)) * np.random.uniform(1, 5)
        elif component_type == 'log':
            return np.log(x + np.random.uniform(20, 30)) * np.random.uniform(4, 8)
        elif component_type == 'step':
            epsilon = 30
            valid_positions = set(range(100, 900))
            for peak in self.changes_positions:
                invalid_range = set(range(peak - epsilon, peak + epsilon + 1))
                valid_positions -= invalid_range
            random_peak_idx = np.random.choice(list(valid_positions))
            self.changes_positions.append(random_peak_idx)
            return self.step_function(x, random_peak_idx, np.random.uniform(15, 30)*(np.random.randint(0,1)*2-1))

    def step_function(self, x, jump_point, jump_height):
        return np.where(x >= jump_point, jump_height, 0)
    
    def generate_peaks(self, x):
        num_peaks = np.random.randint(3, 8)
        peaks_positions = np.sort(np.random.choice(x, num_peaks, replace=False))
        peaks = np.zeros_like(x)
        for pos in peaks_positions:
            peak_width = np.random.uniform(0.5, 1.5)
            peak_amplitude = np.random.uniform(0.5, 2)
            peaks += peak_amplitude * np.exp(-((x - pos) ** 2) / (2 * peak_width ** 2))
        return peaks * 10, peaks_positions.astype(np.int32) - 1
    
    def generate_noise(self, x):
        noise = np.random.default_rng(1).normal(0, self.noise_std, x.size) * 2
        return noise

    def generate_one_curve(self):
        x = np.linspace(1, 1000, self.num_points)

        peaks, peaks_positions = self.generate_peaks(x)
        self.changes_positions.extend(peaks_positions)
        true_baseline = self.generate_trend(x)
        noise = self.generate_noise(x)

        y = peaks + true_baseline + noise

        return x, y, true_baseline, peaks

    def plot_signal(self):
        x, y, true_baseline, peaks = self.generate_one_curve()
        plt.subplot(2, 1, 1)
        plt.plot(x, y, label='Signal', c='b')
        plt.plot(x, true_baseline, label='True Baseline', c='orange')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(x, peaks, label='Peaks', c='r')
        plt.legend()
        plt.show()

    def save_samples_to_json(self, filename):
        x = np.linspace(1, 1000, self.num_points)
        signals = []

        for _ in range(self.num_samples):
            self.changes_positions = []
            peaks, peaks_positions = self.generate_peaks(x)
            self.changes_positions.extend(peaks_positions)
            true_baseline = self.generate_trend(x)
            noise = self.generate_noise(x)
            peaks_encoded = np.zeros_like(x)
            peaks_encoded[peaks_positions] = 1
            y = peaks + true_baseline + noise
            signals.append({'data': list(y), 'trend': list(true_baseline), 'peaks': list(peaks_encoded)})

        with open(filename, "w") as f:
            json.dump(signals, f)


if __name__ == "__main__":
    signal_generator = SignalGenerator(num_samples=200)
    signal_generator.save_samples_to_json("200_generated_signals.json")
    # signal_generator.plot_signal()
