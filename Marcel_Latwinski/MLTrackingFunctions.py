import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import sys
from skimage.feature import peak_local_max
import scipy.stats
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
from scipy.ndimage import gaussian_filter

N = 30 #Plot limiting variable
minMom = 0.7 #min momentum fraction of max momentum
epsilon = 1e-4  # small number to avoid divide-by-near-zero

class particleTracker():
    def __init__(self, name="Particle ", par_num_detectors = None, par_max_momentum = None, par_phiMax = None, par_A = None):

        ### Particle Properties ###
        self.name = str(name) # Name 
        self.q = np.random.choice([-1,1]) # Charge 
        self.max_momentum = par_max_momentum
        self.min_momentum = minMom * par_max_momentum 
        self.p = np.random.uniform(self.min_momentum, self.max_momentum) # Momentum 
        self.A = par_A
        self.C = (self.q * self.A) / self.p # A = Curvature Consant (=3*10^-4 GeV mm^-1)

        ### Detector Properties ###
        self.num_detectors = int(par_num_detectors) # Number of Detectors
        self.detector_pos = np.arange(0, par_num_detectors)
        self.phiMax = par_phiMax
        self.phi0 = np.random.uniform(0, par_phiMax) # Azimuthal Angle

        self.detected_points = []
        self.hough_lines = []
        self.hough_phi = None
        self.hough_r = None

    def particle_path(self, noise = None):
        phis = self.phi0 + self.detector_pos * self.C + scipy.stats.norm.rvs(loc = 0, scale = noise)
        
        x_vals = self.detector_pos * np.cos(phis)
        y_vals = self.detector_pos * np.sin(phis)
        mask = (x_vals >= 0) & (y_vals >= 0)
        #print("X VALUES: ", x_vals, " AND Y VALUES: ", y_vals)
        self.detected_points = np.column_stack((x_vals[mask], y_vals[mask]))

    def plot_path(self, measurement_error=None, color = None):
        if self.detected_points.size == 0:
            return
        xy = self.detected_points
        plt.scatter(xy[:,0], xy[:,1], s=3, color=color)
        plt.plot(xy[:,0], xy[:,1], "--", linewidth=1, color=color)

    def hough_data(self, hough_phi_bins = None, hough_r_bins = None):
        self.hough_phi = np.linspace(0, self.phiMax, hough_phi_bins)
        self.hough_r = np.linspace(-self.A * N / (self.min_momentum), self.A * N / (self.min_momentum), hough_r_bins)
        accumulator = np.zeros((len(self.hough_phi), len(self.hough_r)))

        

        r_min = -self.A * N / self.min_momentum
        r_max = +self.A * N / self.min_momentum

        # Precompute bin width for r
        r0        = self.hough_r[0]
        bin_width = self.hough_r[1] - r0

        phi_list = []
        r_list = []

        for x, y in self.detected_points[1:]:
            houghR = []
            #print("DETECTED POINTS: ",x, " AND ",y)
            for i_phi, phi in enumerate(self.hough_phi):
                denom = 2 * (x * np.cos(phi) - y * np.sin(phi))
                houghRTrue = 1 / ((x**2 + y**2) / denom)

                houghR.append(houghRTrue)

                if houghRTrue < r_min or houghRTrue > r_max:
                    continue

                i_r = np.searchsorted(self.hough_r, houghRTrue) -1  # adjust for bin
                if 0 <= i_r < len(self.hough_r):
                    accumulator[i_phi, i_r] += 1
                    phi_list.append(phi)
                    r_list.append(houghRTrue)


            self.hough_lines.append((np.array(phi_list), np.array(r_list)))
        return accumulator

class Simulation():
    def __init__(self, particle_num = None, max_momentum = None, num_detectors = None, measurement_error = None, phiMax = np.pi/2, A=None):
        print("----------------------------------------")
        print("Starting Simulation...")

        self.num = particle_num
        self.particles = []
        self.phiMax = phiMax
        self.num_detectors = num_detectors
        self.measurement_error = measurement_error
        self.A = A
        self.max_momentum = max_momentum
        self.min_momentum = minMom * max_momentum
        self.accumulator = []

        self.hough_phi = None
        self.hough_r = None

        self.initialPaths = []
        self.peak_positions = []

        Updater = 0

        for i in range(particle_num):
            if i == Updater*5:
                print("COMPLETED ", i, " PARTICLES")
                Updater = Updater + 1
            err = np.random.uniform(-measurement_error, measurement_error)
            
            particle = particleTracker(name = i, par_num_detectors = num_detectors, par_max_momentum = max_momentum, par_phiMax = phiMax, par_A = A)
            particle.particle_path(noise=err)
            self.particles.append(particle)

            self.initialPaths.append((particle.phi0, particle.C))

        self.initialPaths.sort(key=lambda x: x[0])

    def detectors(self):
        angles = np.linspace(0,self.phiMax,500)
        for r in np.arange(0, self.num_detectors):
            x_circle = r * np.cos(angles)
            y_circle = r * np.sin(angles)
            plt.plot(x_circle, y_circle, color = "black", linewidth = 0.5)

    def plot_all_paths(self):
        plt.figure(figsize=(8, 8))

        # Draw detector layers (arcs)
        self.detectors()

        # Create a color map with evenly spaced colors
        cmap = plt.get_cmap("viridis")  # or 'plasma', 'turbo', etc.
        colors = [cmap(i / self.num) for i in range(self.num)]

        # Plot all particle paths
        for i, particle in enumerate(self.particles):
            particle.plot_path(measurement_error = self.measurement_error, color = colors[i] )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Simulated Particle Tracks")
        plt.axis("equal")
        plt.show()

    def hough_transform(self, phi_bins = None, r_bins = None):
        for i, particle in enumerate(self.particles):
            if i == 0:
                self.accumulator = particle.hough_data(hough_phi_bins = phi_bins, hough_r_bins = r_bins)
                self.hough_phi = particle.hough_phi
                self.hough_r = particle.hough_r           
            else:
                self.accumulator += particle.hough_data(hough_phi_bins = phi_bins, hough_r_bins = r_bins)


    def plot_hough_paths(self):
        plt.figure(figsize=(8, 8))

        for particle in self.particles:
            for phi_array, r_array in particle.hough_lines:
                plt.plot(phi_array, r_array, color="black", linewidth=0.5)

        plt.ylabel(r"Inverse Radius $\left(\frac{1}{r}\right)$")
        plt.xlabel("Angle φ")
        plt.title("Hough Transform (φ vs 1/r)")
        plt.ylim(self.hough_r[0], self.hough_r[-1] )
        plt.xlim(self.hough_phi[0], self.hough_phi[-1])
        plt.show()

    def plot_hough_histogram(self, show_peaks=None):

        plt.figure(figsize=(10, 6))
        plt.imshow(self.accumulator.T, extent=[(self.hough_phi[0]), (self.hough_phi[-1]), (self.hough_r[0]), (self.hough_r[-1]) ], 
                    aspect='auto', origin='lower', cmap='hot')
        plt.xlabel('Ejection angle φ (radians)')
        plt.ylabel(r"Inverse Radius $\left(\frac{1}{r}\right)$")
        plt.title('Hough Space (Votes in (φ, r))')
        plt.colorbar(label='Counts')

        if show_peaks:
            peaks = self.find_hough_peaks(sigma=1.0, threshold_percentile=99.95, min_dist=20)
            phi_vals = peaks[:, 0]
            r_vals = peaks[:, 1]
            plt.scatter(phi_vals, r_vals, s=30, color='cyan', marker='o', label='Detected Peaks')
            plt.legend()

        plt.show()

    def find_hough_peaks(self, sigma=None, threshold_percentile=None, min_dist=None):
        smoothed = gaussian_filter(self.accumulator, sigma=sigma)
        threshold = np.percentile(smoothed, threshold_percentile)

        coordinates = peak_local_max(
            smoothed,
            min_distance=min_dist,
            threshold_abs=threshold,
        )

        for row, col in coordinates:
            phi_val = self.hough_phi[row]
            r_val   = self.hough_r[col]
            self.peak_positions.append((phi_val, r_val))
            print(f"Peak: φ = {phi_val:.2f}, r = {r_val:.4f} (Radius = {1/r_val:.2f})")

        self.peak_positions.sort(key=lambda x: x[0])
        return np.array(self.peak_positions)
    
    def ErrorVerification(self):
        
    # Get true values
        true_phi, true_qAp = zip(*self.initialPaths)
        true_phi = np.pi/2 - np.array(true_phi)
        true_qAp = -np.array(true_qAp)

            # Get detected peak positions
        peaks = np.array(self.peak_positions)
        peak_phi = peaks[:, 0]
        peak_qAp = peaks[:, 1]

            # Plot for visual verification
        plt.figure(figsize=(10, 6))
        plt.scatter(true_phi, true_qAp, color='lime', marker='x', label='True (φ₀, qA/p)', s=40)
        plt.scatter(peak_phi, peak_qAp, color='cyan', marker='o', label='Detected Peaks', s=40)
        plt.xlabel('Azimuthal Angle φ (radians)')
        plt.ylabel(r"Inverse Radius $qA/p$ ($1/r$)")
        plt.title('True vs Detected Particle Tracks in Hough Space')
        plt.legend()
        plt.grid(True)
        plt.show()