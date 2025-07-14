<<<<<<< HEAD
print("hello world, edited")
=======
import numpy as np
import matplotlib.pyplot as plt

numTracks = 5
maxR = 10
numDetectors = 8
numDetect = np.linspace(2,numDetectors+1,numDetectors)
#print(numDetect)

##### RANDOM TRACK MECHANISM #####

#Figure and x stuff for curves
plt.figure(figsize=(8,6))
rphiValues = []
detectedPoints = []

for i in range(numTracks):
    #Generate random curve coefficients
    r = np.random.uniform(2, maxR)
    phi = np.random.uniform(0, np.pi/2)
    value = np.random.randint(0, 2) #give either 0 or 1

    rphiValues.append([r,phi])

    # Circle center in global coordinates:
    cx = -r * np.sin(phi)
    cy =  r * np.cos(phi)

    # Generate angle range for the arc (start near phi)
    theta = np.linspace(0, 2*np.pi, 5000)

    # Parametrize circular arc
    if value == 0:
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
    elif value == 1:
        y = cx + r * np.cos(theta)
        x = cy + r * np.sin(theta)
    else:
        print("Problem when printing")

    mask = (x >= 0) & (y >= 0)
    x = x[mask]
    y = y[mask]


    #Only plot if theres something to show
    if len(x) > 0:
        plt.plot(x, y, label=f"r={r:.1f}, ϕ={np.degrees(phi):.1f}°")

    k = 0
    
    for i in range(len(x)):
        r_point = x[i]**2 + y[i]**2
        if np.isclose(r_point,numDetect[k]**2,atol=0.2):
            #print(i)
            if k <= len(numDetect)-2:
                k = k + 1
            detectedPoints.append([x[i],y[i]])

#print("Detected Points are ", detectedPoints)

#axhlines for the detectors
for x in range(2, numDetectors + 2):
    circle = plt.Circle((0, 0), x, color='black', fill=False, linestyle='--', linewidth=0.5)
    plt.gca().add_patch(circle)

print(detectedPoints)
detectedPoints = np.array(detectedPoints)
print(detectedPoints)
plt.scatter(detectedPoints[:, 0],detectedPoints[:, 1])

#Curve plot amnenties:
plt.gca().set_aspect('equal')
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Random Tracks from Particle Production")
plt.show()

###### CURVATURE VALUES #####



>>>>>>> 714fd59 (Attempting to add particle tracking on my paths)
