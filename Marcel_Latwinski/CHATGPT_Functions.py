import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

# --- JIT-compiled path generator ---
@njit
def generate_path(phi0, detector_pos, C, noise_std):
    phis = phi0 + detector_pos * C + np.random.normal(0, noise_std, size=detector_pos.size)
    x_vals = detector_pos * np.cos(phis)
    y_vals = detector_pos * np.sin(phis)
    mask = (x_vals >= 0) & (y_vals >= 0)
    return np.column_stack((x_vals[mask], y_vals[mask]))


class particleTracker():
    def __init__(self, name, par_num_detectors, par_momentum, par_phiMax, par_A, noise_std):

        self.name = str(name)
        self.q = np.random.choice([-1, 1])
        self.p = np.random.uniform(0, par_momentum)
        self.C = (self.q * par_A) / self.p

        self.num_detectors = int(par_num_detectors)
        self.detector_pos = np.arange(self.num_detectors)
        self.phi0 = np.random.uniform(0, par_phiMax)

        self.detected_points = generate_path(self.phi0, self.detector_pos, self.C, noise_std)

    def plot_path(self, color=None):
        if self.detected_points.size == 0:
            return
        xy = self.detected_points
        plt.scatter(xy[:, 0], xy[:, 1], s=3, color=color)
        plt.plot(xy[:, 0], xy[:, 1], "--", linewidth=1, color=color)


class Simulation():
    def __init__(self, particle_num, max_momentum, num_detectors, measurement_error, phiMax=np.pi/2, A=1):
        print("----------------------------------------")
        print("Starting Simulation...")

        self.num = particle_num
        self.phiMax = phiMax
        self.num_detectors = num_detectors
        self.measurement_error = measurement_error
        self.particles = []

        # Parallel particle creation
        def create_particle(i):
            p = np.random.uniform(0.8 * max_momentum, max_momentum)
            return particleTracker(
                name=i,
                par_num_detectors=num_detectors,
                par_momentum=p,
                par_phiMax=phiMax,
                par_A=A,
                noise_std=measurement_error,
            )

        with ThreadPoolExecutor() as executor:
            self.particles = list(executor.map(create_particle, range(particle_num)))

    def detectors(self):
        angles = np.linspace(0, self.phiMax, 500)
        for r in np.arange(0, self.num_detectors):
            x_circle = r * np.cos(angles)
            y_circle = r * np.sin(angles)
            plt.plot(x_circle, y_circle, color="black", linewidth=0.5)

    def plot(self):
        plt.figure(figsize=(8, 8))
        self.detectors()

        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / self.num) for i in range(self.num)]

        for i, particle in enumerate(self.particles):
            particle.plot_path(color=colors[i])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Simulated Particle Tracks")
        plt.axis("equal")
        plt.show()