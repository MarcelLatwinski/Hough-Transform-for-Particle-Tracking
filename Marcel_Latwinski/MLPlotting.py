from MLTrackingFunctions import particleTracker, Simulation
import numpy as np


print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
sim = Simulation(particle_num=10, max_momentum=0.1, measurement_error = 0, num_detectors=9, A = 3*10**(-4))
sim.plot_all_paths()
sim.hough_transform(phi_bins = 1024, r_bins = 512)
sim.plot_hough_paths()
sim.plot_hough_histogram(show_peaks=True, sigma = 0.3, threshold_percentile = 99.8, min_dist = 7);
sim.getPeakImages()
sim.falsePeakDetector(phi_threshold=0.05, qAp_threshold=0.008, printRes = True)
sim.ErrorVerification()
sim.plot_hough_histogram(show_peaks=True, sigma = 0.3, threshold_percentile = 99.8, min_dist = 7);
sim.getPeakImages(save=False)