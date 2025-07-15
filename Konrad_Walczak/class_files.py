import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 

def polar_to_cartesian(r, phi):
    '''
    Function transforming polar coordinates (r, phi) to cartesian (x,y). 
    '''
    return (r*np.sin(phi), r*np.cos(phi))

class Single_Particle_Tracking():

    def __init__(self, name = "DEFAULT", layers = 9):
        '''
        Constructor function for class Single_Particle_Tracking, particle can be given a custom name via |name| (to be used later 
        for massive MC simulations), amount of layers of detectors can be specified via |layers|. 

        Variables in function are:
        self.name - type: string, name of a particle 
        self.q - type: int, randomly assigned electric charge of a particle
        self.layers - type: int, amount of detectors where particle's position is measured
        self.params - type: int, parameters influencing particle's behaviour (mass, velocity, magnetic field etc.) #TODO: currently assumed to be of unity
        self.theta0 - type: float, initial angle of emission of a particle, randomly generated value from an interval of (0, PI/4)
        self.r - type: np.array [1, 2, 3 ... 9] of len(layers), measurements of particle's radial position in polar coordinates at equally spaced detectors
        self.phi - type: np.array of len(layers) (by default set to None), measurments of particle's angular position in polar coordinated at equally spaced detectors
        
        Additional comment:
        Currently it is assumed that we work in 2D cross-section of a detector and particles can only travel in XY dimensions.
        #TODO: if enough time is left then we might expand our model for 3D paths
        
        '''

        self.name = str(name)
        self.q = np.random.choice([-1, 1]) 
        self.layers = int(layers)
        self.params = 1 #TODO
        self.theta0 = np.random.uniform(0, np.pi/4)
        self.r = np.linspace(1, layers, layers)
        self.phi = None

    def generate_path(self, noise_level = 0, max_divisor = 16):
        '''
        Function used to randomize particle's path in the detector. Amount of noise at the detector can be specified via |noise_level|,
        whereas |max_divisor| is used for generation of a deterministic particle's path. This function currently modifies the values of self.phi.

        Firstly, a number from uniform distribution from interval (4, max_divisor) is scaled via self.params, this is angular_divisor.
        Secondly, a np.array of len(self.layers) of equally spaced angles between (0, PI/angular_divisor) and np.array of measurement_noise 
        (from a normal distribution of mean = 0, scale = noise_level; noise is assumed to be of a normal distribution as stated in Chapter 4.2 of [1]) are created.
        Lastly, depending on a particle's charge, both are either added or subtracted from the particle's initial angle, thus obtaining a path in the detector.

        Addition comment:
        Currently I believe this function is sufficient enough to generate simplified circular paths. However, this shall not remain unmodified and should be revised soon.
        '''

        
        angular_divisor = self.params * scipy.stats.uniform.rvs(4, max_divisor)
        dphi = np.linspace(0, np.pi/angular_divisor, self.layers)
        measurement_noise = scipy.stats.norm.rvs(loc = noise_level, size = self.layers)

        if self.q>0:
            self.phi = self.theta0 + dphi + measurement_noisenoise
        else:
            self.phi = self.theta0 - dphi + measurement_noisenoise

    def plot_path(self):
        '''
        Simple function to plot particle's path using (r, phi) in polar coordinates.
        #TODO: redo the style of a plot.
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(self.phi, self.r, marker = '.', color = 'red', label = f"{self.name}")
        plt.title("Path")
        plt.legend()
        #plt.show()
        plt.savefig("simulation_" + self.name)

    def hough_transform(self):
        '''
        Function responsible for Hough transform of a particle's path.

        Addition comment:
        My current understanding of Hough transform is not sufficient to elaborate any further on how it works.

        #TODO: work out what exactly is Hough transform, ask the supervisor and rework the function.
        '''
        x, y = polar_to_cartesian(self.r, self.phi)
        x_axis = np.linspace(0, np.pi, 1000)
        y_axis = []
        for i in range(self.layers):
            curve = 2/self.r[i] * np.sin(x_axis + self.phi[i])
            y_axis.append(curve)
        return x_axis, y_axis

    def plot_hough_transform(self):
        '''
        Simple function to plot Hough transform of a particle's path obtained from hough_transform(self). It remains adequate
        with data format used in an aforementioned function.

        #TODO: redo the function upon modifications of hough_transform()
        '''
        x, y_curves = self.hough_transform()
        plt.figure()
        for i in range(self.layers):
            plt.plot(x, y_curves[i], label = f"", linestyle = '-')
        plt.title("Hough space")
        plt.legend()
        plt.show()
        
'''
BIBLIOGRAPHY:
[1]: Rudolf Frühwirth Are Strandlie "Pattern Recognition, Tracking and Vertex Reconstruction in Particle Detectors"
'''
