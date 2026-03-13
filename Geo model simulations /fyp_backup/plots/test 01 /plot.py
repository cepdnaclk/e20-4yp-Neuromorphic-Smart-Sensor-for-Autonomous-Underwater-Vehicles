import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load txt file
data = pd.read_csv("simulation_log.txt")

# Convert time column
data['time'] = pd.to_datetime(data['time'])

# -------- X position graph --------
plt.figure()
plt.plot(data['time'], data['true_x'], label='True X')
plt.plot(data['time'], data['est_x'], '--', label='Estimated X')
plt.xlabel("Time")
plt.ylabel("X Position")
plt.title("True vs Estimated X")
plt.legend()
plt.grid()
plt.show()

# -------- Y position graph --------
plt.figure()
plt.plot(data['time'], data['true_y'], label='True Y')
plt.plot(data['time'], data['est_y'], '--', label='Estimated Y')
plt.xlabel("Time")
plt.ylabel("Y Position")
plt.title("True vs Estimated Y")
plt.legend()
plt.grid()
plt.show()

# -------- Z position graph --------
plt.figure()
plt.plot(data['time'], data['true_z'], label='True Z')
plt.plot(data['time'], data['est_z'], '--', label='Estimated Z')
plt.xlabel("Time")
plt.ylabel("Depth (Z)")
plt.title("True vs Estimated Z")
plt.legend()
plt.grid()
plt.show()

# -------- Error graph --------
plt.figure()
plt.plot(data['time'], data['error'])
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Localization Error")
plt.grid()
plt.show()

# -------- 3D Trajectory --------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(data['true_x'], data['true_y'], data['true_z'], label="True Path")
ax.plot(data['est_x'], data['est_y'], data['est_z'], '--', label="Estimated Path")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z Depth")
ax.set_title("Submarine Trajectory")

ax.legend()
plt.show()