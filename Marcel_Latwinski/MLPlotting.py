from MLTrackingFunctions import particleTracker, Simulation
import numpy as np


print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
sim = Simulation(particle_num=50, max_momentum=0.1, measurement_error = 0, num_detectors=9, A = 3*10**(-4))
#sim.plot_all_paths()
sim.hough_transform(phi_bins = 2048, r_bins = 1024)
#sim.find_hough_peaks(sigma = 0.6, threshold_percentile = 99.97, min_dist = 10)
#sim.plot_hough_paths()
sim.plot_hough_histogram(show_peaks=True, sigma = 0.6, threshold_percentile = 99.98, min_dist = 10)
sim.getPeakImages()
sim.falsePeakDetector(phi_threshold=0.04, qAp_threshold=0.005)
sim.ErrorVerification()