import numpy as np
import os
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from MLTrackingFunctions import particleTracker, Simulation
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

##### PREPARING THE DATA FOR KERAS #####

#Combining all the peak data
def load_peak_patches(directory):
    patches = []
    filenames = sorted(os.listdir(directory))
    
    for filename in filenames:
        if filename.endswith('.npy'):
            path = os.path.join(directory, filename)
            patch = np.load(path)
            #print(f"Loaded {filename} with shape {patch.shape}")
            patches.append(patch)
    
    # Try to convert to array
    try:
        arr = np.array(patches)
        #print("All patches stacked into array with shape:", arr.shape)
        return arr
    except ValueError as e:
        print("Error converting patches list to array:", e)
        return patches  # Return list for now if error

# Pre-allocate empty lists to accumulate data
x_all = []
y_all = []

for i in range(400):
    sim = Simulation(particle_num=50, max_momentum=0.1, measurement_error = 0, num_detectors=9, A = 3*10**(-4))
    sim.hough_transform(phi_bins = 1024, r_bins = 1024)
    sim.find_hough_peaks(sigma = 1, threshold_percentile = 99.9, min_dist = 10)
    sim.getPeakImages()
    sim.falsePeakDetector(phi_threshold=0.04, qAp_threshold=0.01)
    if i % 50 == 0:
        print("Finished", i, "Loop")

    #peak_patches = load_peak_patches("Marcel_Latwinski/PeakImages")

    #Adding grayscale axis for data array for keras
    #Preparing the array of peak data (min-max Normalisation)
    x = np.array(sim.peak_patches)[..., np.newaxis].astype(np.float32)

    #Preparing labels for data (whats a true and fake peak)
    #labels_path = "Marcel_Latwinski/detection_labels.npy"
    #detection_labels = np.load(labels_path)  # shape: (N,)
    #print("Loaded labels with shape:", detection_labels.shape)
    y = np.array(sim.detection_labels).astype(np.float32)  # Binary labels

    #print("X SHAPE: ", x.shape[0])
    #print("Y SHAPE: ", y.shape[0])

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} samples at iteration {i}")

    x_all.append(x)
    y_all.append(y)

    #print("X ALL SHAPE: ", x_all.shape[0])
    #print("Y ALL SHAPE: ", y_all.shape[0])

# Combine all batches into one large array
x_all = np.concatenate(x_all, axis=0)  # shape becomes (5200, 64, 64, 1)
y_all = np.concatenate(y_all, axis=0)  # shape becomes (5200,)

mean = x_all.mean()
std = x_all.std()
x_all = (x_all - mean) / std

#Splitting data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print("Training set shape:", x_train.shape)
print("Validation set shape:", x_test.shape)


####### BUILDING KERAS MODEL #######

input_shape = (64, 64, 1)  # Since input is grayscale

model = keras.Sequential([
    keras.Input(shape=input_shape),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss="binary_crossentropy",  # binary classification
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(buffer_size=10000).batch(64).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_ds, epochs=10, validation_data=test_ds, verbose=2)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Accuracy on primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(history.history['accuracy'], label='Train Accuracy', color=color, linestyle='-')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color=color, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)

# Loss on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(history.history['loss'], label='Train Loss', color=color, linestyle='-')
ax2.plot(history.history['val_loss'], label='Val Loss', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)

# Add legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='center right')

plt.title('Training and Validation Accuracy & Loss')
plt.tight_layout()
plt.show()
