# Hybrid Neural Network & Hough Transform Particle Tracking

**Authors:** Marcel Latwinski, Konrad Walczak  
**Supervisor:** Dr. hab. Marcin Wolter  
**Language:** Python  

## Overview

This project implements a **hybrid particle tracking algorithm** combining the **Hough Transform** with a **Neural Network**. It simulates charged particle tracks, maps detector hits into Hough space (`φ` vs `1/r`), and uses a neural network to validate peaks for accurate track reconstruction.

## Features

- Simulates charged particle trajectories with variable momentum and noise.
- Hough Transform to identify candidate tracks.
- Neural network for peak validation to reduce false positives.
- Visualization of both particle paths and Hough space histograms.
- Modular Python code for easy parameter adjustments.

## Usage

```python
from MLTrackingFunctions import Simulation

sim = Simulation(particle_num=10, max_momentum=5, num_detectors=50, measurement_error=0.01, phiMax=np.pi/2, A=3e-4)
sim.plot_all_paths()
sim.plot_hough_paths(phi_bins=256, r_bins=256)
sim.plot_hough_histogram()
