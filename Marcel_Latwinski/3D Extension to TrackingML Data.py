import numpy as np
import matplotlib.pyplot as plt
import math
#from MLTrackingFunctions import Simulation, particleTracker

class ThreeDParticleTracker:
    def __init__(self, name="Particle"):

        ### Particle Properties ###
        self.name = str(name) # Name
        self.detected_points = []
        self.slices = []

        #self.x_min, self.x_max, self.y_min, self.y_max = None

    def generate_helical_track(self, noise = None):
        # Helix parameters
        r_final = 5           # radius of the helix in meters
        pitch = 4         # distance per 2π turn along z-axis
        turns = 0.1             # number of full turns
        points_per_turn = 1000
        spiral_factor = 1

        expected_kappa = 1/r_final
        self.expected_kappa = expected_kappa

        # Parameterize the helix
        theta = np.linspace(0, 2 * np.pi * turns, int(points_per_turn * turns))
        z = pitch * theta / (2 * np.pi) + np.random.normal(0, noise, size=theta.shape)

        r = spiral_factor * z

        # Ensure final radius matches r_final
        if turns > 0:
            r *= r_final/r.max()

        x = r * np.cos(theta) + np.random.normal(0, noise, size=theta.shape)
        y = r * np.sin(theta) + np.random.normal(0, noise, size=theta.shape)
        

        # Stack into array
        helix_points = np.column_stack((x, y, z))
        self.detected_points = helix_points
        # Output array of shape (N, 3)
        print("helix_points.shape:", helix_points.shape)

        # Optional: plot to verify
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(x, y, z, label='Helical Track')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def Track_Flatten(self, thickness = None, overlap = None):
        assert overlap < thickness, "Overlap must be smaller than thickness"

        z_values = self.detected_points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        start_z = z_min

        while start_z < z_max:
            end_z = start_z + thickness
            mask = (z_values >= start_z) & (z_values < end_z)
            slice_points = self.detected_points[mask]
            self.slices.append(slice_points)
            start_z += (thickness - overlap)

        print(f"Generated {len(self.slices)} slices along z-axis.")
    
    def Plot_Flattened_paths(self):
        if not hasattr(self, 'slices'):
            print("Error: self.slices not found. Make sure you've sliced the data first.")
            return

        # Select first 2 and last 2 slices
        num_slices = len(self.slices)
        selected_indices = [0, 1]  # First two slices
        if num_slices > 2:
            selected_indices.extend([num_slices-2, num_slices-1])  # Last two slices

        # Create figure with appropriate number of subplots
        num_plots = len(selected_indices)
        cols = min(4, num_plots)  # Max 4 columns
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if num_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Make it iterable even for single plot

        # Compute global x and y limits across all slices (using only selected slices)
        selected_points = np.vstack([self.slices[i] for i in selected_indices if len(self.slices[i]) > 0])
        all_points = np.vstack([s for s in self.slices if len(s) > 0])
        if len(selected_points) > 0:
            x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        else:
            x_min, x_max, y_min, y_max = -1, 1, -1, 1  # Default values if no points

        for plot_idx, slice_idx in enumerate(selected_indices):
            ax = axes[plot_idx]
            slice_points = self.slices[slice_idx]

            if len(slice_points) > 0:
                ax.scatter(slice_points[:, 0], slice_points[:, 1], s=10, alpha=0.6)
                ax.set_title(f"Slice {slice_idx} (N={len(slice_points)})")
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_aspect('equal')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Hide unused axes if any
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def hough_data(self, points, hough_phi_bins=None, hough_kappa_bins=None):
        A = 3e-4  # Constant for 2T field
        q = 1     # Assume unit charge

        x = points[:, 0]
        y = points[:, 1]

        #Cartesian to Polar
        phi_h = np.arctan2(y, x)
        r_h = np.sqrt(x**2 + y**2)

        # Define Hough parameter space
        phi_t_min, phi_t_max = -np.pi, np.pi
        expected_kappa = 1/r_h.mean()
        kappa_min = -2*expected_kappa
        kappa_max = 2*expected_kappa

        self.phi_t_bins = np.linspace(phi_t_min, phi_t_max, hough_phi_bins)
        self.kappa_bins = np.linspace(kappa_min, kappa_max, hough_kappa_bins)

        #Create dense bins for accurate curve sampling
        dense_factor = 10  # How much denser than the coarse grid
        phi_t_dense = np.linspace(-np.pi, np.pi, hough_phi_bins * dense_factor)

        accumulator = np.zeros((hough_phi_bins, hough_kappa_bins))

        for i in range(len(points)):
            rh = r_h[i]
            ph = phi_h[i]

            # Calculate kappa on dense grid
            kappa_dense = np.sin(phi_t_dense - ph) / rh

            # Filter valid kappa values
            valid_mask = (kappa_dense >= kappa_min) & (kappa_dense <= kappa_max)
            phi_t_valid = phi_t_dense[valid_mask]
            kappa_valid = kappa_dense[valid_mask]

            # Map dense samples back to coarse bins
            phi_t_indices = np.searchsorted(self.phi_t_bins, phi_t_valid, side='right') - 1
            kappa_indices = np.searchsorted(self.kappa_bins, kappa_valid, side='right') - 1

            # Remove any invalid indices
            valid_idx = (
                (phi_t_indices >= 0) & (phi_t_indices < hough_phi_bins) &
                (kappa_indices >= 0) & (kappa_indices < hough_kappa_bins))

            phi_t_indices = phi_t_indices[valid_idx]
            kappa_indices = kappa_indices[valid_idx]

            # Use unique bin pairs to avoid double-counting
            unique_bin_pairs = set(zip(phi_t_indices, kappa_indices))

            # Create temporary accumulator for this point
            temp_acc = np.zeros_like(accumulator)
            for phi_idx, kappa_idx in unique_bin_pairs:
                temp_acc[phi_idx, kappa_idx] = 1  # One vote per unique bin

            accumulator += temp_acc
    
        return accumulator
  
    def visualize_hough_lines_from_hits(self, points, hough_phi_bins=1000, ax=None):
        """
        For visual understanding only: plot curves in (φ_t, κappa) space
        from a few individual hits.
        """
        x = points[:, 0]
        y = points[:, 1]
        phi_h = np.arctan2(y, x)
        r_h = np.sqrt(x**2 + y**2)

        phi_t_vals = np.linspace(-np.pi, np.pi, hough_phi_bins)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the curves for the first few hits
        for i in range(min(5, len(points))):  # Only 5 to avoid crowding
            kappa_curve = np.sin(phi_t_vals - phi_h[i]) / r_h[i]
            ax.plot(phi_t_vals, kappa_curve, label=f"Hit {i}")

        M = 10
        ax.set_xlabel("φ_t (track azimuthal angle)")
        ax.set_ylabel("κappa = (φ_t - φ_hit)/r")
        ax.set_ylim(M*-0.02, M*0.02)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_title("Hough curves from hits")
        ax.grid(True)

    
    def plot_accumulator(self, hough_phi_bins=None, hough_kappa_bins=None, visualise=None):
        if not hasattr(self, 'slices') or len(self.slices) == 0:
            print("Error: no slices found. Run Track_Flatten first.")
            return

        # Select first 2 and last 2 slices
        num_slices = len(self.slices)
        selected_indices = [0, 1]  # First two slices
        if num_slices > 2:
            selected_indices.extend([num_slices-2, num_slices-1])  # Last two slices

        # Create figure with appropriate number of subplots
        num_plots = len(selected_indices)
        cols = min(4, num_plots)  # Max 4 columns
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if num_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Make it iterable even for single plot

        for plot_idx, slice_idx in enumerate(selected_indices):
            ax = axes[plot_idx]
            slice_points = self.slices[slice_idx]

            if len(slice_points) > 0:
                points_2d = slice_points[:, :2]
                if visualise:
                    self.visualize_hough_lines_from_hits(points=points_2d, hough_phi_bins=1000, ax=ax)
                else:
                    acc = self.hough_data(points=points_2d, 
                                        hough_phi_bins=hough_phi_bins, 
                                        hough_kappa_bins=hough_kappa_bins)
                    print(f"Slice {slice_idx} max votes:", acc.max())

                    im = ax.imshow(
                        acc.T,
                        extent=[-np.pi, np.pi, self.kappa_bins[0], self.kappa_bins[-1]],
                        origin='lower',
                        aspect='auto',
                        cmap='inferno'
                    )
                    ax.set_title(f"Slice {slice_idx} (N={len(slice_points)})")
                    ax.set_xlabel("phi_t")
                    ax.set_ylabel("kappa (qA/pT)")
                    ax.axhline(self.expected_kappa, color='cyan', linestyle='--', alpha=0.5)
                    ax.axhline(-self.expected_kappa, color='cyan', linestyle='--', alpha=0.5)
                    plt.colorbar(im, ax=ax)

        # Hide unused axes if any
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    #def peak_finder(self):
        


particle = ThreeDParticleTracker(name="particle")
particle.generate_helical_track(noise = 0.001)
particle.Track_Flatten(thickness = 0.1, overlap = 0.02)
particle.Plot_Flattened_paths()
particle.plot_accumulator(hough_phi_bins=256, hough_kappa_bins=256, visualise = False)

