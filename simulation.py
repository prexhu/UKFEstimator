import numpy as np

def generate_truth(T, dt):
    # Generates a complex trajectory with multiple maneuvers
    x = np.zeros((T, 4))
    x[0] = [0, 0, 5, 2] # Initial pos and vel
    
    # Define trajectory phases: (duration, turn_rate, acceleration)
    phases = [
        (200, 0.0, 0.0),    # Straight
        (150, 0.05, 0.0),   # Left turn
        (100, 0.0, 0.1),    # Acceleration
        (200, -0.04, 0.0),  # Right turn
        (150, 0.08, -0.05), # Sharp left + deceleration
        (200, 0.0, 0.0)     # Final straight
    ]
    
    k = 1
    for duration, omega, acc in phases:
        for _ in range(duration):
            if k >= T: break
            
            vx, vy = x[k-1, 2], x[k-1, 3]
            v = np.sqrt(vx**2 + vy**2)
            heading = np.arctan2(vy, vx)
            
            # Apply maneuvers
            new_v = v + acc * dt
            new_heading = heading + omega * dt
            
            x[k, 2] = new_v * np.cos(new_heading)
            x[k, 3] = new_v * np.sin(new_heading)
            x[k, 0] = x[k-1, 0] + x[k, 2] * dt
            x[k, 1] = x[k-1, 1] + x[k, 3] * dt
            k += 1
            
    return x[:T]

def generate_measurements(x, sensor_pos, R):
    z = []
    for xi in x:
        dx = xi[0] - sensor_pos[0]
        dy = xi[1] - sensor_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        noise = np.random.multivariate_normal(np.zeros(2), R)
        z.append(np.array([r, theta]) + noise)
    return np.array(z)