from class_file import Single_Particle, Simulation
from numpy import pi
a = Simulation(n = 100, detector_angle=pi/4)
#a.plot_paths()
a.hough_transform_histogram(hough_space_size= 512, log_scale = False)
a.hough_transform_histogram(hough_space_size=1024, log_scale = True)