import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

def create_ellipse_plot(output_path="conceptual_ellipse_metrics.pdf"):
    """
    Creates a compact, standalone plot for the elliptical uncertainty region and its metrics.
    """
    fig, ax = plt.subplots(figsize=(6, 6)) # Square figure for compact layout

    # --- Define Parameters ---
    true_params = np.array([0, 0])
    center_ellipse = np.array([-0.01, 0.005])
    
    angle_rad = np.deg2rad(30)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    cov_matrix = rotation_matrix @ np.diag([0.08**2, 0.03**2]) @ rotation_matrix.T
    
    # --- Calculate Geometry and Metrics ---
    eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix)
    order = eigenvalues_cov.argsort()[::-1]
    eigenvalues_cov, eigenvectors_cov = eigenvalues_cov[order], eigenvectors_cov[:, order]

    d_max_val = np.sqrt(eigenvalues_cov[0])
    d_min_val = np.sqrt(eigenvalues_cov[1])
    d_max_vec = eigenvectors_cov[:, 0]
    d_min_vec = eigenvectors_cov[:, 1]
    
    delta_a_max = np.sqrt(cov_matrix[0, 0])
    delta_b_max = np.sqrt(cov_matrix[1, 1])

    # --- Plotting ---
    # Faint semi-axes
    p1 = center_ellipse + d_max_val * d_max_vec
    p2 = center_ellipse - d_max_val * d_max_vec
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linestyle=':', lw=1.5)
    p3 = center_ellipse + d_min_val * d_min_vec
    p4 = center_ellipse - d_min_val * d_min_vec
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color='lightgray', linestyle=':', lw=1.5)

    # Ellipse shape
    ellipse_fill = Ellipse(xy=center_ellipse, width=2*d_max_val, height=2*d_min_val, angle=np.degrees(np.arctan2(d_max_vec[1], d_max_vec[0])),
                           facecolor='mediumpurple', alpha=0.3)
    ax.add_patch(ellipse_fill)
    ellipse_border = Ellipse(xy=center_ellipse, width=2*d_max_val, height=2*d_min_val, angle=np.degrees(np.arctan2(d_max_vec[1], d_max_vec[0])),
                             edgecolor='purple', facecolor='none', lw=2, label='Elliptical Confidence Region')
    ax.add_patch(ellipse_border)
    
    # Bounding Box for Axis-Aligned Deviations
    bbox = Rectangle(xy=center_ellipse - np.array([delta_a_max, delta_b_max]),
                     width=2*delta_a_max, height=2*delta_b_max, edgecolor='black', facecolor='none',
                     linestyle='--', lw=1.5, label=r'Axis-Deviations ($\Delta A_{\mathrm{max}}, \Delta B_{\mathrm{max}}$)')
    ax.add_patch(bbox)

    # Worst-Case Deviation
    end_point = center_ellipse + d_max_val * d_max_vec
    ax.plot([center_ellipse[0], end_point[0]], [center_ellipse[1], end_point[1]],
            color='purple', linestyle='-', lw=2, label=r'$d_{\mathrm{max}}$')

    # Points
    ax.plot(true_params[0], true_params[1], 'x', color='red', ms=5, mew=2.5, label=r'True Parameters $\theta$')
    ax.plot(center_ellipse[0], center_ellipse[1], 'P', color='purple', ms=9, label=r'Estimate $\hat{\theta}_{ell}$')
    
    # --- Finalize ---
    ax.set_title('Metrics for Elliptical Region')
    ax.set_xlabel(r'Parameter $a$'); ax.set_ylabel(r'Parameter $b$')
    ax.legend(loc='upper right'); ax.grid(True); ax.axis('equal')
    ax.set_xlim(-0.12, 0.12); ax.set_ylim(-0.12, 0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Ellipse metrics plot saved to: {output_path}")


def create_rectangle_plot(output_path="conceptual_rectangle_metrics.pdf"):
    """
    Creates a compact, standalone plot for the rectangular uncertainty region and its metrics.
    """
    fig, ax = plt.subplots(figsize=(6, 6)) # Square figure for compact layout

    # --- Define Parameters ---
    true_params = np.array([0, 0])
    center_rect = np.array([-0.008, 0.008])
    epsilons = {'a': 0.065, 'b': 0.05}

    # --- Calculate Metrics ---
    d_max_rect_val = np.sqrt(epsilons['a']**2 + epsilons['b']**2)
    
    # --- Plotting ---
    # Rectangle Shape
    rect_fill = Rectangle(xy=center_rect - np.array([epsilons['a'], epsilons['b']]),
                          width=2 * epsilons['a'], height=2 * epsilons['b'],
                          facecolor='green', alpha=0.2)
    ax.add_patch(rect_fill)
    rect_border = Rectangle(xy=center_rect - np.array([epsilons['a'], epsilons['b']]),
                            width=2 * epsilons['a'], height=2 * epsilons['b'],
                            edgecolor='darkgreen', facecolor='none', lw=2, linestyle='--', label='Rectangular Confidence Region')
    ax.add_patch(rect_border)
    
    # Axis-Aligned Deviations
    ax.plot([center_rect[0], center_rect[0] + epsilons['a']], [center_rect[1], center_rect[1]],
            color='dimgray', linestyle=':', lw=2, label=r'$\Delta A_{\mathrm{max}} = \epsilon_A$')
    ax.plot([center_rect[0], center_rect[0]], [center_rect[1], center_rect[1] + epsilons['b']],
            color='gray', linestyle=':', lw=2, label=r'$\Delta B_{\mathrm{max}} = \epsilon_B$')

    # Worst-Case Deviation
    end_point = center_rect + np.array([epsilons['a'], epsilons['b']])
    ax.plot([center_rect[0], end_point[0]], [center_rect[1], end_point[1]],
            color='green', linestyle='--', lw=2.5, label=r'$d_{\mathrm{max}}$')
    
    # Points
    ax.plot(true_params[0], true_params[1], 'x', color='red', ms=5, mew=2.5, label=r'True Parameters $\theta$')
    ax.plot(center_rect[0], center_rect[1], 's', color='green', ms=5, label=r'Estimate $\hat{\theta}_{rect}$')

    # --- Finalize ---
    ax.set_title('Metrics for Rectangular Region')
    ax.set_xlabel(r'Parameter $a$'); ax.set_ylabel(r'Parameter $b$')
    ax.legend(loc='upper right'); ax.grid(False); ax.axis('equal')
    ax.set_xlim(-0.12, 0.12); ax.set_ylim(-0.12, 0.12)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Rectangle metrics plot saved to: {output_path}")


if __name__ == '__main__':
    create_ellipse_plot(output_path="conceptual_ellipse_metrics.pdf")
    create_rectangle_plot(output_path="conceptual_rectangle_metrics.pdf")