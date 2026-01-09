import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ukf import UKF
from models import motion_model, radar_model
from simulation import generate_truth, generate_measurements
from state_fusion import covariance_intersection

# Simulation setting
dt = 0.1
T = 1000
# generate ground truth trajectory
truth = generate_truth(T, dt)

# Tow Lidar different locations
# TODO: adjust positions from config file
sensor1 = np.array([150, 600])
sensor2 = np.array([200, 100])

# Noise: Sensor 1 is better at range, Sensor 2 is better at bearing (hypothetically)
R1 = np.diag([2.0**2, (2.0*np.pi/180)**2])
R2 = np.diag([5.0**2, (0.5*np.pi/180)**2])

# Adding noise 
q_val = 2.0 
Q = q_val * np.array([
    [dt**3/3, 0, dt**2/2, 0],
    [0, dt**3/3, 0, dt**2/2],
    [dt**2/2, 0, dt, 0],
    [0, dt**2/2, 0, dt]
])

z1 = generate_measurements(truth, sensor1, R1)
z2 = generate_measurements(truth, sensor2, R2)

# UKF
ukf1 = UKF(4, 2, motion_model, lambda x: radar_model(x, sensor1), Q, R1, dt)
ukf2 = UKF(4, 2, motion_model, lambda x: radar_model(x, sensor2), Q, R2, dt)

# Set UKF parameters
for u in [ukf1, ukf2]:
    u.alpha = 0.1
    u.x = truth[0] + np.random.normal(0, 2, 4)
    u.P = np.eye(4) * 20

est1, est2, est_fused, omega_history = [], [], [], []

for k in range(T):
    ukf1.predict()
    ukf1.update(z1[k])
    
    ukf2.predict()
    ukf2.update(z2[k])
    
    xf, Pf, w = covariance_intersection(ukf1.x, ukf1.P, ukf2.x, ukf2.P)
    
    est1.append(ukf1.x.copy())
    est2.append(ukf2.x.copy())
    est_fused.append(xf)
    omega_history.append(w)

est1, est2, est_f = np.array(est1), np.array(est2), np.array(est_fused)

# Calulate RMSE
def calc_rmse(est, true):
    return np.sqrt(np.mean(np.sum((est[:, :2] - true[:, :2])**2, axis=1)))

rmse1 = calc_rmse(est1, truth)
rmse2 = calc_rmse(est2, truth)
rmse_f = calc_rmse(est_f, truth)

print(f"RMSE Sensor 1: {rmse1:.4f}")
print(f"RMSE Sensor 2: {rmse2:.4f}")
print(f"RMSE Fused:    {rmse_f:.4f}")

# Plot and data export
df = pd.DataFrame({
    'truth_x': truth[:,0], 'truth_y': truth[:,1],
    'est1_x': est1[:,0], 'est1_y': est1[:,1],
    'est2_x': est2[:,0], 'est2_y': est2[:,1],
    'fused_x': est_f[:,0], 'fused_y': est_f[:,1],
    'omega': omega_history
})
df.to_csv('tracking_results.csv', index=False)

plt.figure(figsize=(12, 8))
plt.plot(truth[:,0], truth[:,1], 'k-', label='Ground Truth', linewidth=2)
plt.plot(est1[:,0], est1[:,1], 'b--', alpha=0.5, label=f'Sensor 1 (RMSE={rmse1:.2f})')
plt.plot(est2[:,0], est2[:,1], 'g--', alpha=0.5, label=f'Sensor 2 (RMSE={rmse2:.2f})')
plt.plot(est_f[:,0], est_f[:,1], 'r-', label=f'Fused (CI) (RMSE={rmse_f:.2f})', linewidth=1.5)
plt.scatter([sensor1[0], sensor2[0]], [sensor1[1], sensor2[1]], c=['b', 'g'], marker='X', s=100)
plt.title('Multi-Sensor Target Tracking')
plt.legend()
plt.grid(True)
plt.savefig('final_trajectory.png')

plt.figure(figsize=(10, 4))
plt.plot(omega_history, 'm', label='Fusion Weight (Omega)')
plt.ylabel('Weight (Omega)')
plt.xlabel('Time Step')
plt.title('Dynamic Fusion Weight over Time')
plt.grid(True)
plt.savefig('omega_plot.png')