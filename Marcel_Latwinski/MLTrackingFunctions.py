import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.feature import peak_local_max
import scipy.stats
from scipy.ndimage import gaussian_filter, center_of_mass
import os
from scipy.spatial import distance_matrix
import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import tensorflow as tf
import zipfile
from sklearn.model_selection import train_test_split

N = 10 #Plot limiting variable
minMom = 0.5 #min momentum fraction of max momentum
epsilon = 1e-4  # small number to avoid divide-by-near-zero

def getImage(image, size, x_center, y_center):
    half = size // 2
    y_min = y_center - half
    y_max = y_center + half
    x_min = x_center - half
    x_max = x_center + half

    pad_top = max(0, -y_min)
    pad_left = max(0, -x_min)
    pad_bottom = max(0, y_max - image.shape[0])
    pad_right = max(0, x_max - image.shape[1])

    # Clamp the coordinates inside the image
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)

    patch = image[y_min:y_max, x_min:x_max]

    # Pad if needed to get full size
    patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return patch

def save_patch_high_res(patch, filepath, cmap='hot', dpi=300):
    if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
        print(f"Warning: patch is empty, skipping save for {filepath}")
        return

    h, w = patch.shape
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(patch, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

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

    def hough_data(self, hough_phi_bins=None, hough_r_bins=None):
        self.hough_phi = np.linspace(0, self.phiMax, hough_phi_bins)
        self.hough_r = np.linspace(-self.A * N / self.min_momentum,
                                    self.A * N / self.min_momentum, hough_r_bins)

        accumulator_clipped = np.zeros((len(self.hough_phi), len(self.hough_r)))
        hough_phi_dense = np.linspace(self.hough_phi[0], self.hough_phi[-1], 10 * len(self.hough_phi))

        r_min = self.hough_r[0]
        r_max = self.hough_r[-1]

        hough_phi = self.hough_phi
        hough_r = self.hough_r

        cos_phi = np.cos(self.hough_phi)
        sin_phi = np.sin(self.hough_phi)

        cos_phi_dense = np.cos(hough_phi_dense)
        sin_phi_dense = np.sin(hough_phi_dense)

        self.hough_lines = []

        # Loop over all detected points (excluding origin)
        for x, y in self.detected_points[1:]:
            denom_dense = 2 * (-x * sin_phi_dense + y * cos_phi_dense)
            hough_r_dense = denom_dense / (x**2 + y**2)

            # Filter valid indices on dense grid
            valid_mask = (
              (hough_r_dense >= r_min) & (hough_r_dense <= r_max) &
              np.isfinite(hough_r_dense)
            )

            phi_valid = hough_phi_dense[valid_mask]
            r_valid = hough_r_dense[valid_mask]

            # Convert to bin indices on original coarse grid (hough_phi, hough_r)
            i_phi = np.searchsorted(hough_phi, phi_valid, side='right') - 1
            i_r = np.searchsorted(hough_r, r_valid, side='right') - 1

            # Remove any invalid indices
            valid_idx = (
                (i_phi >= 0) & (i_phi < len(self.hough_phi)) &
                (i_r >= 0) & (i_r < len(self.hough_r))
            )

            i_phi = i_phi[valid_idx]
            i_r   = i_r[valid_idx]

            # Unique pairs of (i_phi, i_r) -> prevent double-counting from dense sampling
            unique_bin_indices = set(zip(i_phi, i_r))

            # Create a temporary zero array and set 1s for unique bin hits
            temp_acc = np.zeros_like(accumulator_clipped)
            for phi_idx, r_idx in unique_bin_indices:
               temp_acc[phi_idx, r_idx] = 1  # Only one vote per point per bin

            accumulator_clipped += temp_acc

            self.hough_lines.append((phi_valid[valid_idx], r_valid[valid_idx]))

        return accumulator_clipped

class Simulation():
    def __init__(self, particle_num = None, max_momentum = None, num_detectors = None, measurement_error = None, phiMax = np.pi/2, A=None):
        #print("----------------------------------------")
        #print("Starting Simulation...")

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
        self.peak_coordinates = None
        self.peak_patches = []
        self.detection_labels = None
        self.matched_peak_pairs = []

        Updater = 0

        for i in range(particle_num):
            if i == Updater*5:
                #print("COMPLETED ", i, " PARTICLES")
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

    def plot_hough_histogram(self, show_peaks=None, sigma = None, threshold_percentile = None, min_dist = None, CNNpeaks = None):

        plt.figure(figsize=(14, 6))
        plt.imshow(self.accumulator.T, extent=[(self.hough_phi[0]), (self.hough_phi[-1]), (self.hough_r[0]), (self.hough_r[-1]) ],
                    aspect='auto', origin='lower', cmap='hot')
        plt.xlabel('Ejection angle φ (radians)')
        plt.ylabel(r"Inverse Radius $\left(\frac{1}{r}\right)$")
        plt.title('Hough Space (Votes in (φ, r))')
        plt.colorbar(label='Counts')

        if show_peaks:
            if CNNpeaks is not None and len(CNNpeaks) > 0:
                cnn_phi = CNNpeaks[:, 0]
                cnn_r = CNNpeaks[:, 1]
                plt.scatter(cnn_phi, cnn_r, s=50, color='magenta', marker='x', label='CNN Peaks')
            else:
                peaks = self.find_hough_peaks(sigma=sigma, threshold_percentile=threshold_percentile, min_dist=min_dist)
                phi_vals = peaks[:, 0]
                r_vals   = peaks[:, 1]

                # Check if detection_labels exist and match length
                if self.detection_labels is not None and len(self.detection_labels) == len(peaks):
                    for i in range(len(peaks)):
                        color = 'lime' if self.detection_labels[i] == 1 else 'dodgerblue'
                        label = 'Matched Peak' if self.detection_labels[i] == 1 else 'Unmatched Peak'
                        plt.scatter(phi_vals[i], r_vals[i], s=30, color=color, marker='o', label=label if i == 0 or (self.detection_labels[:i] != self.detection_labels[i]).all() else "")

                else:
                    # Default behavior if no labels
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

        window_size = 5
        pad = window_size // 2

        refined_peaks = []
        for y, x in coordinates:
            if pad <= y < smoothed.shape[0] - pad and pad <= x < smoothed.shape[1] - pad:
                window = smoothed[y-pad:y+pad+1, x-pad:x+pad+1]
                dy, dx = center_of_mass(window)
                refined_y = y - pad + dy
                refined_x = x - pad + dx

                # Convert to Hough space coordinates
                phi_idx = int(round(refined_y))
                r_idx = int(round(refined_x))

                # Clamp to valid range
                if 0 <= phi_idx < len(self.hough_phi) and 0 <= r_idx < len(self.hough_r):
                    phi_val = self.hough_phi[phi_idx]
                    r_val   = self.hough_r[r_idx]
                    refined_peaks.append((refined_y, refined_x, phi_val, r_val))

        # Sort by phi value
        refined_peaks.sort(key=lambda x: x[2])

        self.peak_coordinates = np.array([(y, x) for y, x, phi, r in refined_peaks])
        self.peak_positions = [(phi, r) for y, x, phi, r in refined_peaks]

        return np.array(self.peak_positions)

    def ErrorVerification(self):

        # Get true values
        true_phi, true_qAp = zip(*self.initialPaths)
        true_phi = np.array(true_phi)
        true_qAp = np.array(true_qAp)

        #print("I should be seeing on my plot points with values: ", true_phi, " and ", true_qAp)

        #for i in range(len(true_phi)):
        #    print(f" --- TRUE PEAKS {i}: φ = {true_phi[i]:.4f}, qA/p = {true_qAp[i]:.4f}")


        # Get detected peak positions

        detected_peaks = np.array(self.peak_positions)
        #peak_phi = np.pi/2-detected_peaks[:, 0]
        peak_phi = detected_peaks[:, 0]
        peak_qAp = detected_peaks[:, 1]

        # Plot for visual verification
        plt.figure(figsize=(10, 6))

        plt.scatter(true_phi, true_qAp, color='lime', marker='x', label='True (φ₀, qA/p)', s=40)

        #Plot lines connecting matches peaks
        if hasattr(self, 'matched_peak_pairs'):
            for (true_phi, true_qAp, det_phi, det_qAp) in self.matched_peak_pairs:
                plt.plot([true_phi, det_phi], [true_qAp, det_qAp], color='orange', linestyle='--', linewidth=1, alpha=0.7)

        plt.scatter(peak_phi, peak_qAp, color='cyan', marker='o', label='Detected Peaks', s=40)
        plt.xlabel('Azimuthal Angle φ (radians)')
        plt.ylabel(r"Inverse Radius $qA/p$ ($1/r$)")
        plt.title('True vs Detected Particle Tracks in Hough Space')
        plt.legend()
        plt.grid(True)
        plt.show()

    def falsePeakDetector(self, phi_threshold=None, qAp_threshold=None, printRes = None):
        # Get true values
        true_phi, true_qAp = zip(*self.initialPaths)
        true_phi = np.array(true_phi)
        true_qAp = np.array(true_qAp)
        true_peaks = np.column_stack((true_phi, true_qAp))

        # Get detected peak positions
        detected_peaks = np.array(self.peak_positions)
        #detected_peaks[:, 0] = np.pi/2 - detected_peaks[:, 0]

        # To store best matches as (true_idx, detected_idx, distance)
        best_matches = []

        for j, (true_phi_val, true_qAp_val) in enumerate(true_peaks):
            d_phi = np.abs(detected_peaks[:, 0] - true_phi_val)
            d_qAp = np.abs(detected_peaks[:, 1] - true_qAp_val)
            within_threshold_mask = (d_phi <= phi_threshold) & (d_qAp <= qAp_threshold)
            candidate_indices = np.where(within_threshold_mask)[0]

            if len(candidate_indices) == 0:
                continue

            distances = np.sqrt(d_phi[candidate_indices]**2 + d_qAp[candidate_indices]**2)
            min_idx = candidate_indices[np.argmin(distances)]
            min_distance = distances[np.argmin(distances)]
            best_matches.append((j, min_idx, min_distance))

        # Sort and assign matches uniquely
        best_matches.sort(key=lambda x: x[2])
        final_true_matched = set()
        final_detected_matched = set()

        for true_idx, detected_idx, dist in best_matches:
            if true_idx not in final_true_matched and detected_idx not in final_detected_matched:
                final_true_matched.add(true_idx)
                final_detected_matched.add(detected_idx)

                # Store line segment: (true_phi, true_qAp, detected_phi, detected_qAp)
                self.matched_peak_pairs.append((
                    true_peaks[true_idx][0], true_peaks[true_idx][1],  # True
                    detected_peaks[detected_idx][0], detected_peaks[detected_idx][1]  # Detected
                ))

        # Count results
        true_positives = len(final_detected_matched)
        false_positives = len(detected_peaks) - true_positives
        false_negatives = len(true_peaks) - len(final_true_matched)
        if printRes == True:
          print(f"✅ Matched Peaks: {true_positives}")
          print(f"❌ Falsely Detected Peaks: {false_positives}")
          print(f"❌ Missed True Peaks: {false_negatives}")

        # Label detected peaks
        labels = np.zeros(len(detected_peaks), dtype=int)
        for i in final_detected_matched:
            labels[i] = 1

        self.detection_labels = labels
        #np.save(r"C:\Users\Marce\Hough-Transform-for-Particle-Tracking\Marcel_Latwinski\detection_labels.npy", self.detection_labels)


    def getPeakImages(self, size = 64, save_dir="Marcel_Latwinski/PeakImages", zip_name="PeakImages.zip", verbose = False, save=None):
        if not hasattr(self, 'peak_coordinates') or len(self.peak_coordinates) == 0:
            print("No peak coordinates found. Run `find_hough_peaks()` first.")
            return

        if os.path.exists(save_dir):
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Couldn't delete {file_path}: {e}")
        else:
            os.makedirs(save_dir)

        for idx, (phi_i, r_i) in enumerate(self.peak_coordinates):
            # Treat phi_i as the row index (y), r_i as the column index (x)
            phi_i = int(phi_i)
            r_i = int(r_i)

            patch = getImage(self.accumulator, size=size, x_center=r_i, y_center=phi_i)

            self.peak_patches.append(patch)
            if save==True:
                npy_path = os.path.join(save_dir, f"peak_{idx}.npy")
                np.save(npy_path, patch)

            #jpeg_path = os.path.join(save_dir, f"peak_{idx}.png")
            #save_patch_high_res(patch, jpeg_path, cmap='hot')

            if verbose:
                print(f"Saved peak {idx} at (row={phi_i}, col={r_i}) to {npy_path}") # and {jpeg_path}")

                # Zip the folder
                zip_path = os.path.join(save_dir, "..", zip_name)  # Save zip one level above the folder
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for filename in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, filename)
                        zipf.write(file_path, arcname=filename)  # arcname keeps filenames clean in the zip
                if verbose:
                    print(f"\n✅ All peak images saved and zipped to: {zip_path}")