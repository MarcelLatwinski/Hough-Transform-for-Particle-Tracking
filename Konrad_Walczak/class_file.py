import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
def empty_line():
    print(20*" ")
    print(20*"*")

class Single_Particle():
    def __init__(self, name = "Particle_", layers = 9, theta0_min = 0, theta0_max = np.pi, detector_angle = np.pi/4, pt = 10):
        #initiation conditions
        self.theta0_min = theta0_min
        self.theta0_max = theta0_max

        #detector's properties
        self.pT = pt
        self.layers = int(layers)
        self.A = 3*1e-4
        self.detector_angle = detector_angle/2

        #particle's properties
        self.name = name
        self.q = np.random.choice([-1,1])
        self.theta0 = np.random.uniform(theta0_min + 0.05 * (theta0_max-theta0_min), theta0_max - (theta0_max-theta0_min))
        self.r = np.linspace(1, layers, layers)
        self.phi = None
        self.maximal_curvature = self.q * self.A / self.pT

        #
        self.detection_points_x, self.detection_points_y, self.curvature = None, None, None
        

    def generate_path(self, measurement_error = 0):
        curvature = np.random.uniform(0, self.maximal_curvature)
        detection_points_x, detection_points_y = np.zeros(shape = self.layers), np.zeros(shape = self.layers)
        thetas = self.theta0 + scipy.stats.norm.rvs(loc = 0, scale = measurement_error) + self.r * curvature
        detection_points_x = self.r * np.cos(thetas)
        detection_points_y = self.r * np.sin(thetas)
        mask = np.abs(thetas)<= 2* self.detector_angle 
        self.detection_points_x, self.detection_points_y, self.curvature = detection_points_x[mask], detection_points_y[mask], curvature
    
    def plot_path(self, measurement_error = 0):
        self.generate_path(measurement_error)
        detection_points_x, detection_points_y, curvature = self.detection_points_x, self.detection_points_y, self.curvature
        angles = np.linspace(-np.pi/2, np.pi/2, 1000)

        plt.figure(figsize= (10, 10))
        plt.axhline(color = 'black', linewidth = 0.5)
        for radius in self.r:
            x_circle = radius * np.cos(angles)
            y_circle = radius * np.sin(angles)
            plt.plot(x_circle, y_circle, color = "black", linewidth = 0.5)
        plt.plot(detection_points_x, detection_points_y, color = "red", marker = 'o', linestyle = 'dashed', label = {self.name})
        
        
        plt.legend()
        plt.show()


    def perform_hough_transform(self, hough_space_bins=1024):

        # Parameter ranges
        theta_range = np.linspace(self.theta0_min, self.theta0_max, hough_space_bins)
        curvature_range = np.linspace(-self.maximal_curvature, self.maximal_curvature, hough_space_bins)

        
        plt.figure(figsize=(8, 8))
        for x, y in zip(self.detection_points_x, self.detection_points_y):    
            curvature_line = np.sin((np.arctan2(y, x) - theta_range)) / np.sqrt(x**2 + y**2)
            
            
            #curvature_line = np.clip(curvature_line, curvature_range[0], curvature_range[-1])
            plt.plot(theta_range, curvature_line, alpha=0.6, color = 'black')

        plt.xlabel("Theta₀ (rad)")
        plt.ylabel("Curvature")
        plt.title(f"Hough Space Lines for: {self.name}")
        plt.grid(True)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.show()

    def hough_transform_data(self, hough_space_bins = 1024):
        theta_range = np.linspace(self.theta0_min, self.theta0_max, hough_space_bins)
        curvature_range = np.linspace(-self.maximal_curvature, self.maximal_curvature, hough_space_bins)
        curvature_lines = []

        for x, y in zip(self.detection_points_x, self.detection_points_y):    
            curvature_lines.append(np.sin((np.arctan2(y, x) - theta_range)) / np.sqrt(x**2 + y**2))
        
        return theta_range, curvature_lines

class Simulation():
    def __init__(self, n = 5, layers = 9, theta0_min = 0, theta0_max = np.pi, detector_angle = np.pi/4, pt_max = 10):
        empty_line()
        print("INITIATING SIMULATION...")

        self.layers = layers
        self.theta0_min = theta0_min
        self.theta0_max = theta0_max
        self.detector_angle = detector_angle
        self.pt = pt_max
        self.curvature_max = 1/self.pt
        
        
        self.count = n
        self.particles = []
        for i in range(n):
            pt = np.random.uniform(0, pt_max)
            measurement_error = np.random.uniform(0, np.pi/100)

            particle = Single_Particle(name = i, layers = self.layers, theta0_min = self.theta0_min, theta0_max = self.theta0_max, detector_angle = self.detector_angle, pt = pt)

            particle.generate_path(measurement_error)
            self.particles.append(particle)
            
    def plot_paths(self):
        empty_line()
        print("PLOTTING PATHS...")
        fig, ax = plt.subplots(figsize= (10, 10))
        angles = np.linspace(-np.pi/2, np.pi/2, 1000)
        plt.axhline(color = 'black', linewidth = 0.5)

        for radius in self.particles[0].r:
            x_circle = radius * np.cos(angles)
            y_circle = radius * np.sin(angles)
            ax.plot(x_circle, y_circle, color = "black", linewidth = 0.5)
        for i in range(self.count):
            #ax.plot(self.particles[i].detection_points_x, self.particles[i].detection_points_y, marker = 'o', linestyle = 'dashed', label = {self.particles[i].name})
            ax.plot(self.particles[i].detection_points_x, self.particles[i].detection_points_y, marker = 'o', linestyle = 'dashed', linewidth = 2)
        
        max_r = self.particles[0].r[-1] + 1  
        for angle in [self.detector_angle, -self.detector_angle]:
            x_line = [0, max_r * np.cos(angle)]
            y_line = [0, max_r * np.sin(angle)]
            plt.plot(x_line, y_line, color="black", linestyle="--", linewidth=5, label = f"Limit of the detector at {np.rad2deg(angle):.1f} degrees")
        
        plt.plot(0,0, markersize = '50', color = 'gold', marker = '*')
        plt.title("Collisions inside the detector")
        plt.legend()
        plt.show()


    def hough_transform_lines(self):

        plt.figure(figsize=(8, 8))

        for particle in self.particles:
            theta_range, curvature_lines = particle.hough_transform_data()
            for curve in curvature_lines:
                plt.plot(theta_range, curve, alpha=0.6, color = 'black')
        
        plt.xlabel("Theta₀ (rad)")
        plt.ylabel("Curvature")
        plt.grid(True)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.show()

    def hough_transform_histogram(self ,hough_space_size = 256, log_scale = False):
        empty_line()
        print("PLOTTING HISTOGRAM OF HOUGH TRANSFORM..")
        
        theta_range, curvature_range = np.linspace(self.theta0_min, self.theta0_max, hough_space_size), np.linspace(-self.curvature_max, self.curvature_max, hough_space_size)
        accumulator = np.zeros((hough_space_size, hough_space_size))

        for particle in self.particles:
            theta_values, curves = particle.hough_transform_data()
            for curve_values in curves:
                for theta_val, curve_val in zip(theta_values, curve_values):
                    theta_index = np.searchsorted(theta_range, theta_val) - 1
                    curve_index = np.searchsorted(curvature_range, curve_val) - 1
                    if 0 <= theta_index < hough_space_size and 0 <= curve_index < hough_space_size:
                        accumulator[curve_index, theta_index] +=1

        plt.figure(figsize = (7,7))
        extent = [self.theta0_min, self.theta0_max, -self.curvature_max, self.curvature_max]
        
        if log_scale:
            plt.imshow(np.log(accumulator), extent = extent, origin = 'lower', aspect = 'auto', cmap = 'inferno')
        else: 
            plt.imshow(accumulator, extent = extent, origin = 'lower', aspect = 'auto', cmap = 'inferno')
        
        plt.grid(True)
        plt.xlabel("Theta₀ (rad)")
        plt.ylabel("Curvature")
        plt.title("2D Hough Transform Histogram")
        plt.show()