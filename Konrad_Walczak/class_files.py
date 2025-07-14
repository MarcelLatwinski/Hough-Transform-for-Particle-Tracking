import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 

def polar_to_cartesian(r, phi):
    '''
    transforms polar coordinates (r, phi) to cartesian (x,y) 
    '''
    return (r*np.sin(phi), r*np.cos(phi))

class Single_Particle_Tracking():

    def __init__(self, name = "DEFAULT", layers = 9):
        '''
        Cons
        
        
        
        '''

        self.name = str(name)
        self.q = np.random.choice([-1, 1]) 
        self.layers = int(layers)
        self.params = 1
        self.r = np.linspace(1, layers, layers)
        self.theta0 = np.random.uniform(0, np.pi/4)
        self.phi = None

    def generate_path(self, noise_level = 0):
        '''
        
        
        
        '''

        max_divisor = 12
        i = self.params * scipy.stats.uniform.rvs(4, max_divisor)
        dphi = np.linspace(0, np.pi/i, self.layers)
        noise = scipy.stats.norm.rvs(scale = noise_level, size = self.layers)

        if self.q>0:
            self.phi = self.theta0 + dphi + noise
        else:
            self.phi = self.theta0 - dphi + noise

    def plot_path(self):
        '''
        
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(self.phi, self.r, marker = '.', color = 'red', label = f"{self.name}")
        plt.title("Path")
        plt.legend()
        plt.savefig("simulation_" + self.name)

    def hough_transform(self):
        '''
        function responsible for Hough transform 
        '''
        x, y = polar_to_cartesian(self.r, self.phi)
        x_axis = np.linspace(0, np.pi, 1000)
        y_axis = []
        for i in range(self.layers):
            curve = 2/self.r[i] * np.sin(x_axis + self.phi[i])
            y_axis.append(curve)
        return x_axis, y_axis

    def plot_hough_transform(self):
        x, y_curves = self.hough_transform()
        plt.figure()
        for i in range(self.layers):
            plt.plot(x, y_curves[i], label = f"", linestyle = '-')
        plt.title("Hough space")
        plt.legend()
        plt.show()
        
