import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.feature import peak_local_max
import scipy.stats
from scipy.ndimage import gaussian_filter, center_of_mass
import os
from scipy.spatial import distance_matrix
import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import tensorflow as tf
import zipfile
from sklearn.model_selection import train_test_split

x_all = []
y_all = []
peak_positions_all = []

for i in range(1000):
    sim = Simulation(particle_num=70, max_momentum=0.1, measurement_error = 0, num_detectors=9, A = 3*10**(-4))
    sim.hough_transform(phi_bins = 1024, r_bins = 512)
    sim.find_hough_peaks(sigma = 0.3, threshold_percentile = 99.775, min_dist = 5)
    sim.getPeakImages(save=False)
    sim.falsePeakDetector(phi_threshold=0.05, qAp_threshold=0.008, printRes = False)
    #sim.plot_hough_histogram(show_peaks=True, sigma = 0.3, threshold_percentile = 99.8, min_dist = 5);
    #sim.ErrorVerification()

    if i % 50 == 0:
        print("Finished", i, "Loop")

    #Preparing the array of peak data (min-max Normalisation)
    x = np.array(sim.peak_patches)#[..., np.newaxis].astype(np.float32)

    #Preparing labels for data (whats a true and fake peak)
    y = np.array(sim.detection_labels).astype(np.float32)  # Binary labels
    '''
    for j in range(1):
        plt.figure(figsize=(4, 3))
        plt.imshow(x[j], cmap='viridis')
        plt.colorbar()
        plt.title(f"2D Array as Image with label: {y[j]}")
        plt.show()
    '''

    positions = np.array(sim.peak_positions)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} samples at iteration {i}")

    x_all.append(x)
    y_all.append(y)
    peak_positions_all.append(positions)

# Combine all batches into one large array
x_all = np.concatenate(x_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
peak_positions_all = np.concatenate(peak_positions_all, axis=0)

#Splitting data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.4, random_state=42, stratify=y_all
)

#Normalise
mean = x_train.mean(); std = x_train.std()
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std

print("Training set shape:", x_train.shape)
print("Validation set shape:", x_test.shape)

print("Overall label counts:", np.unique(y_all, return_counts=True))
print("Train label counts:  ", np.unique(y_train, return_counts=True))
print("Test  label counts:  ", np.unique(y_test, return_counts=True))

true_indices  = np.where(y_all == 1)[0][:]
false_indices = np.where(y_all == 0)[0][:]

#print("TRUE: ", true_indices.shape)
#print(false_indices.shape)

#print(true_indices)
#print(false_indices)

fig, axes = plt.subplots(2, 5, figsize=(20, 10))

max_plots = 5  # because axes has only 5 columns
for i in range(min(len(true_indices),len(false_indices),max_plots)):
    # True patch
    idx_true = true_indices[i]
    #print(idx_true)
    patch_true = x_all[idx_true].squeeze()
    phi_true, r_true = peak_positions_all[idx_true]

    im1 = axes[0, i].imshow(patch_true, cmap='hot', extent=[
        phi_true - 0.5, phi_true + 0.5,
        r_true - 0.5,   r_true + 0.5
    ])
    axes[0, i].set_title(f"True: φ={phi_true:.2f}, r={r_true:.2f}")
    axes[0, i].set_xlabel("φ")
    axes[0, i].set_ylabel("r")
    fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

    # False patch
    idx_false = false_indices[i]
    #print(idx_false)
    patch_false = x_all[idx_false].squeeze()
    phi_false, r_false = peak_positions_all[idx_false]

    im2 = axes[1, i].imshow(patch_false, cmap='hot', extent=[
        phi_false - 0.5, phi_false + 0.5,
        r_false - 0.5,   r_false + 0.5
    ])
    axes[1, i].set_title(f"False: φ={phi_false:.2f}, r={r_false:.2f}")
    axes[1, i].set_xlabel("φ")
    axes[1, i].set_ylabel("r")
    fig.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

############### BUILDING KERAS MODEL ###################

###### BUILDING KERAS MODEL ######
input_shape = (64, 64, 1)  # Grayscale input

model = keras.Sequential([
    keras.Input(shape=input_shape),

    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),  # NEW: Added dropout after each block

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),  # NEW: Increased dropout deeper in the network

    # Block 3 (MODIFIED: Reduced to 128 filters to limit overfitting)
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),  # NEW

    # Block 4 (REMOVED: Original 256-filter layer was likely too large)
    # Flatten and Classify
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),  # MODIFIED: Added L2
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# MODIFIED: Lower learning rate + added ReduceLROnPlateau
optimizer = keras.optimizers.Adam(learning_rate=1e-5)  # Reduced from 1e-4
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

# Data pipeline (unchanged)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(buffer_size=10000).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# NEW: Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)  # Dynamic LR
]

# Train with callbacks
history = model.fit(
    train_ds,
    epochs=30,  # Increased since EarlyStopping will halt early
    validation_data=test_ds,
    callbacks=callbacks,  # NEW
    verbose=2
)

model.save('hough_peak_validator.keras')

# Plotting (unchanged)
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(history.history['accuracy'], label='Train Accuracy', color=color, linestyle='-')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color=color, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(history.history['loss'], label='Train Loss', color=color, linestyle='-')
ax2.plot(history.history['val_loss'], label='Val Loss', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='center right')
plt.title('Training and Validation Accuracy & Loss')
plt.tight_layout()
plt.show()

####### TESTING MODEL ########

sim = Simulation(particle_num=100, max_momentum=0.1, measurement_error = 0, num_detectors=9, A = 3*10**(-4))
sim.hough_transform(phi_bins = 1024, r_bins = 512)
sim.find_hough_peaks(sigma = 0.3, threshold_percentile = 99.78, min_dist = 5)
#sim.falsePeakDetector(phi_threshold=0.05, qAp_threshold=0.008, printRes = False)
sim.plot_hough_histogram(show_peaks=True, sigma = 0.3, threshold_percentile = 99.78, min_dist = 5);
#sim.ErrorVerification()
sim.getPeakImages(save=False)

#Preparing the array of peak data (min-max Normalisation)
x = np.array(sim.peak_patches)#[..., np.newaxis].astype(np.float32)

#Preparing labels for data (whats a true and fake peak)
y = np.array(sim.detection_labels).astype(np.float32)  # Binary labels
positions = np.array(sim.peak_positions)

print("x shape: ", x.shape)
print("y shape: ", y.shape)
print("positions shape: ", positions.shape)

model = keras.models.load_model('hough_peak_validator.h5')

#Predict
predictions = model.predict(x, verbose=0).flatten()  # shape: (N,)

# 4. Define classification threshold (adjust if needed)
threshold = 0.7
true_indices = np.where(predictions > threshold)[0]
#print(predictions)
# 5. Get corresponding positions
true_peak_positions = positions[true_indices]

sim.plot_hough_histogram(show_peaks=True, sigma = 0.3, threshold_percentile = 99.78, min_dist = 7, CNNpeaks = true_peak_positions)
