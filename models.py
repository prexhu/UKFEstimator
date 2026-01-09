import numpy as np

def motion_model(x, dt):
    # Constant Velocity Model
    # x = [px, py, vx, vy]
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])
    return F @ x

def radar_model(x, sensor_pos):
    # Nonlinear Observation: Range and Bearing
    dx = x[0] - sensor_pos[0]
    dy = x[1] - sensor_pos[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return np.array([r, theta])