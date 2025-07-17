
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.morphology import local_maxima
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

numTracks = 1
minR = 60
maxR = 80
numDetectors = 8
numDetect = np.linspace(2,numDetectors+1,numDetectors)
#print(numDetect)

############ RANDOM TRACK AND DETECTION MECHANISM ###########

#Figure and x stuff for curves
plt.figure(figsize=(8,6))
rphiValues = []
detectedPoints = []

for i in range(numTracks):
    #Generate random curve coefficients
    while True:
        r = np.random.uniform(-maxR, maxR)
        if abs(r) >= minR:
            break
    phi = np.random.uniform(0, np.pi/2)

    rphiValues.append([r,phi])

    # Circle center in global coordinates:
    cx = -r * np.sin(phi)
    cy =  r * np.cos(phi)

    # Generate angle range for the arc (start near phi)
    theta = np.linspace(0, 2*np.pi, 10000)

    # Parametrize circular arc
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    mask = (x >= 0) & (y >= 0)
    x = x[mask]
    y = y[mask]


    #Only plot if theres something to show
    if len(x) > 0:
        plt.plot(x, y, label=f"r={r:.1f}, ϕ={(phi):.1f} rad")

    k = 0
    
    detector_hits = set()

    for i in range(len(x)):
        r_point = np.sqrt(x[i]**2 + y[i]**2)
        for j, detector_radius in enumerate(numDetect):
            if j in detector_hits:
                continue  # already found this intersection

            if np.isclose(r_point, detector_radius, atol=0.03):
                detectedPoints.append([x[i], y[i]])
                detector_hits.add(j)
                #print(f"Intersection with detector {detector_radius:.2f} at ({x[i]:.2f}, {y[i]:.2f})")


#print("Detected Points are ", detectedPoints)

#axhlines for the detectors
for x in range(2, numDetectors + 2):
    circle = plt.Circle((0, 0), x, color='black', fill=False, linestyle='--', linewidth=0.5)
    plt.gca().add_patch(circle)

#print(detectedPoints)
detectedPoints = np.array(detectedPoints)
print(len(detectedPoints))
plt.scatter(detectedPoints[:, 0],detectedPoints[:, 1])

#Curve plot amnenties:
plt.gca().set_aspect('equal')
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Random Tracks from Particle Production")

############ HOUGH TRANSFORM ###########

plt.figure(figsize=(10, 6))
houghPhi = np.linspace(0, np.pi/2, 400)
houghR = []

#Accumulator stuff:
houghRValues = np.linspace(-1/maxR - 0.5, 1/maxR + 0.5, 1000)
accumulator = np.zeros((len(houghPhi), len(houghRValues)))

for x, y in detectedPoints:
    houghR = []
    for i_phi, phi in enumerate(houghPhi):
        houghRTrue = 1/( (x**2 + y**2) /  (2*(x*np.sin(phi) - y*np.cos(phi))) ) 
        houghR.append(houghRTrue)
        for i_r, r in enumerate(houghRValues):
            if np.isclose(r, houghRTrue, atol=2*houghRValues.max()/len(houghRValues)):
                accumulator[i_phi, i_r] += 1
    plt.plot(houghPhi, houghR, color = "black")

plt.ylabel(r"Inverse Radius $\left(\frac{1}{r}\right)$")
plt.xlabel("Angle φ")
plt.xlim(0,np.pi/2)
plt.ylim(-1/maxR - 0.5, 1/maxR + 0.5)
plt.title("Hough Transform (φ vs 1/r)")


############ ACCUMULATOR ###########

plt.figure(figsize=(10, 6))
plt.imshow(accumulator.T, extent=[(houghPhi[0]), (houghPhi[-1]), houghRValues[0], houghRValues[-1]], 
           aspect='auto', origin='lower', cmap='hot')
plt.xlabel('Ejection angle φ (radians)')
plt.ylabel(r"Inverse Radius $\left(\frac{1}{r}\right)$")
plt.title('Hough Space (Votes in (φ, r))')
plt.colorbar(label='Counts')


#######FINDING MAX#######

threshold = numDetectors * 0.8
image = (accumulator >= threshold) * accumulator

coordinates = peak_local_max(
    image,               # thresholded accumulator
    min_distance=5,      # minimum distance between peaks (in pixels)
    threshold_abs=threshold,
)

for i, (row, col) in enumerate(coordinates):
    print(f"Peak {i}: phi = {houghPhi[row]:.2f}, R = {houghRValues[col]:.4f}")
    print(f"This gives us RADIUS: {1/houghRValues[col]:.2f} and ANGLE: {houghPhi[row]:.2f}")
plt.show()