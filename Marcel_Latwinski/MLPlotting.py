from MLTrackingFunctions import particleTracker, Simulation
import numpy as np

sim = Simulation(particle_num=10, max_momentum=0.1, measurement_error = 0, num_detectors=9, A = 3*10**(-4))
sim.plot_all_paths()
sim.hough_transform(phi_bins = 1024, r_bins = 1024)
#sim.plot_hough_paths()
sim.plot_hough_histogram(show_peaks=True)
sim.ErrorVerification()
