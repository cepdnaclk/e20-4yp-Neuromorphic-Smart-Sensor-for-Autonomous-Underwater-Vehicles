import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from geomodel_clean import get_auv_position

def plot_auv_geometry():
    # Use the exact test values that proved the model
    LAT_B, LON_B = 7.208300, 79.835800
    alpha = math.radians(15) # East-West tilt at buoy
    mu = math.radians(10)    # North-South tilt at buoy
    beta = math.radians(8)   # East-West tilt at AUV
    eta = math.radians(5)    # North-South tilt at AUV
    L = 100.0                # 100m total cable
    z = 50.0                 # 50m AUV depth

    # Get the calculated coordinates
    results = get_auv_position(LAT_B, LON_B, alpha, mu, beta, eta, L, z)
    
    # Extract local coordinates for 3D plotting
    # Buoy is at Origin (0,0,0)
    buoy_pos = np.array([0, 0, 0])
    
    # Ballast Position (End of Segment 1)
    # Using your math: x_B = l1x * sin(alpha), y_B = l1y * sin(mu), z_B = l1x * cos(alpha)
    ballast_pos = np.array([
        results['l1'] * math.sin(alpha) / math.sqrt(1 + math.tan(mu)**2 * math.cos(alpha)**2), # Approximate X for visual
        results['l1'] * math.sin(mu) / math.sqrt(1 + math.tan(mu)**2 * math.cos(alpha)**2),    # Approximate Y for visual
        -results['depth_B'] # Negative for depth
    ])
    
    # AUV Position
    auv_pos = np.array([
        results['dx_east'],
        results['dy_north'],
        -results['depth_AUV']
    ])

    # Setup 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the tether segments
    ax.plot([buoy_pos[0], ballast_pos[0]], [buoy_pos[1], ballast_pos[1]], [buoy_pos[2], ballast_pos[2]], 
            'k-', linewidth=2, label='Tether Segment 1 (L1)')
    ax.plot([ballast_pos[0], auv_pos[0]], [ballast_pos[1], auv_pos[1]], [ballast_pos[2], auv_pos[2]], 
            'gray', linewidth=2, linestyle='--', label='Tether Segment 2 (L2)')

    # Plot the physical entities
    ax.scatter(*buoy_pos, color='orange', s=200, label='Surface Buoy (RTK-CORS)', edgecolors='black')
    ax.scatter(*ballast_pos, color='black', s=150, marker='s', label=f"Sliding Ballast ({results['depth_B']:.1f}m)")
    ax.scatter(*auv_pos, color='blue', s=200, marker='^', label=f"AUV ({results['depth_AUV']:.1f}m)")

    # Draw vertical drop lines to show depth clearly
    ax.plot([auv_pos[0], auv_pos[0]], [auv_pos[1], auv_pos[1]], [0, auv_pos[2]], 'b:', alpha=0.5)
    ax.plot([ballast_pos[0], ballast_pos[0]], [ballast_pos[1], ballast_pos[1]], [0, ballast_pos[2]], 'k:', alpha=0.5)

    # Formatting
    ax.set_xlabel('East Offset (m)')
    ax.set_ylabel('North Offset (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_title('3D Tether Geometry & AUV Localization\n(Eliminating the need for DVL)')
    
    # Simulate water surface
    xx, yy = np.meshgrid(np.linspace(-5, 25, 2), np.linspace(-5, 25, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.1)

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('auv_localization_3d.png', dpi=300)
    print("Saved 'auv_localization_3d.png' successfully! Drop this into your presentation.")
    plt.show()

if __name__ == "__main__":
    plot_auv_geometry()