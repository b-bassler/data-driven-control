import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage[light]{firasans}
        \usepackage{amsmath}
    """,
})


def create_ellipse_plot(output_path="conceptual_ellipse_metrics.pdf"):
        """
        Creates a compact, standalone plot for the elliptical uncertainty region and its metrics.
        """
        fig, ax = plt.subplots(figsize=(7.5, 7.5)) # Square figure for compact layout

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
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='grey', linestyle=':', lw=1)
        p3 = center_ellipse + d_min_val * d_min_vec
        p4 = center_ellipse - d_min_val * d_min_vec
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color='grey', linestyle=':', lw=1)

        # Ellipse shape
        ellipse_fill = Ellipse(xy=center_ellipse, width=2*d_max_val, height=2*d_min_val, angle=np.degrees(np.arctan2(d_max_vec[1], d_max_vec[0])),
                                facecolor='mediumpurple', alpha=0.3, label='Confidence area')
        ax.add_patch(ellipse_fill)
        ellipse_border = Ellipse(xy=center_ellipse, width=2*d_max_val, height=2*d_min_val, angle=np.degrees(np.arctan2(d_max_vec[1], d_max_vec[0])),
                                edgecolor='purple', facecolor='none', lw=1.5, label='Elliptical Confidence Region')
        ax.add_patch(ellipse_border)

        concept_box = Rectangle(
        xy=center_ellipse,
        width=delta_a_max,
        height=delta_b_max,
        edgecolor='black',
        facecolor='none',
        linestyle='--',  # Bleibt gestrichelt
        lw=1,
        label=r'$\Delta A_{\mathrm{max}}, \Delta B_{\mathrm{max}}$'
        )
        ax.add_patch(concept_box)

        cx, cy = center_ellipse


        ax.plot([cx, cx + delta_a_max], [cy, cy], color='black', linestyle='-', lw=1) # linestyle='-' für durchgezogen

     
        ax.plot([cx, cx], [cy, cy + delta_b_max], color='black', linestyle='-', lw=1) # linestyle='-' für durchgezogen

        ax.text(cx + delta_a_max / 2, cy + 0.003, r'$\Delta A_{\mathrm{max}}$', 
                ha='center', va='bottom', fontsize=18, color='black')

        # delta_b_max Beschriftung: Näher an cx, z.B. cx + 0.01 (leicht rechts von der Linie)
        ax.text(cx + 0.003, cy + delta_b_max / 2, r'$\Delta B_{\mathrm{max}}$', 
                ha='left', va='center', fontsize=18, rotation='vertical', color='black')



                # Worst-Case Deviation
        d_max_vec_positive = np.abs(d_max_vec)

        # Endpunkt berechnen, indem wir den Vektor SUBTRAHIEREN
        end_point = center_ellipse - d_max_val * d_max_vec_positive
        ax.plot([center_ellipse[0], end_point[0]], [center_ellipse[1], end_point[1]],
                color='black', linestyle='-', lw=1) # Label in Legende kann hier weg, da wir es direkt plotten

                # 1. Startposition des Textes (in der Mitte der Linie)
        text_pos_on_line = center_ellipse - (d_max_val / 2) * d_max_vec_positive

        # 2. Orthogonalen Vektor berechnen
        # Ein Vektor (x, y) hat den orthogonalen Vektor (-y, x)
        # Unser Richtungsvektor ist (-d_max_vec_positive[0], -d_max_vec_positive[1])
        perp_vec = np.array([d_max_vec_positive[1], -d_max_vec_positive[0]])

        # 3. Orthogonalen Vektor auf Länge 1 normieren
        # Dies stellt sicher, dass der Abstand immer gleich ist, unabhängig von der Steigung
        if np.linalg.norm(perp_vec) > 0:
                perp_vec_norm = perp_vec / np.linalg.norm(perp_vec)
        else:
                perp_vec_norm = np.array([0, 0]) # Fallback, sollte nicht eintreten

        # 4. Den gewünschten Abstand definieren (diesen Wert kannst du anpassen)
        orthogonal_offset = -0.01  # Der Abstand senkrecht zur Linie in Plot-Einheiten

        # 5. Endgültige Textposition berechnen
        final_text_pos = text_pos_on_line + orthogonal_offset * perp_vec_norm

        # 6. Rotationswinkel der Linie bleibt gleich
        angle_of_line = np.degrees(np.arctan2(-d_max_vec_positive[1], -d_max_vec_positive[0]))

# Dein bisheriger Code zum Platzieren des Texts
        ax.text(final_text_pos[0], final_text_pos[1], r'$d_{\mathrm{max}}$',
        color='black', fontsize=18,
        rotation=angle_of_line + 180,
        ha='center', va='center')


        # Points
        ax.plot(true_params[0], true_params[1], 'x', color='red', ms=5, mew=1, label=r'True Parameters $\theta$')
        ax.plot(center_ellipse[0], center_ellipse[1], '+', color='purple', # Marker zu '+' geändert
        ms=9, mew=1.5, label=r'Center')      # mew hinzugefügt
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
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # --- Define Parameters ---
    true_params = np.array([0, 0])
    center_rect = np.array([-0.008, 0.008])
    epsilons = {'a': 0.065, 'b': 0.05}

    # --- Plotting ---
    # Rectangle Shape
    rect_fill = Rectangle(xy=center_rect - np.array([epsilons['a'], epsilons['b']]),
                          width=2 * epsilons['a'], height=2 * epsilons['b'],
                          facecolor='green', alpha=0.2, label = 'Confidence area')
    ax.add_patch(rect_fill)
    rect_border = Rectangle(xy=center_rect - np.array([epsilons['a'], epsilons['b']]),
                            width=2 * epsilons['a'], height=2 * epsilons['b'],
                            edgecolor='darkgreen', facecolor='none', lw=2, linestyle='-',
                            label='Rectangular Confidence Region')
    ax.add_patch(rect_border)
    
    # Axis-Aligned Deviations
    ax.plot([center_rect[0], center_rect[0] + epsilons['a']], [center_rect[1], center_rect[1]],
            color='black', linestyle='-', lw=1, label=r'$\epsilon_A$')
    ax.plot([center_rect[0], center_rect[0]], [center_rect[1], center_rect[1] + epsilons['b']],
            color='black', linestyle='-', lw=1, label=r'$\epsilon_B$')

    # --- NEU: Beschriftung für Achsenparallele Abweichung ---
    # Position für epsilon_A (mittig, leicht unter der Linie)
    ax.text(center_rect[0] + epsilons['a'] / 2, center_rect[1] - 0.005, r'$\epsilon_A$',
            ha='center', va='top', fontsize=20, color='black')
    # Position für epsilon_B (mittig, leicht links von der Linie)
    ax.text(center_rect[0] - 0.005, center_rect[1] + epsilons['b'] / 2, r'$\epsilon_B$',
            ha='right', va='center', fontsize=20, rotation='vertical', color='black')

    # --- Worst-Case Deviation ---
    wdc_vector = np.array([epsilons['a'], epsilons['b']])
    end_point = center_rect - wdc_vector
    ax.plot([center_rect[0], end_point[0]], [center_rect[1], end_point[1]],
            color='black', linestyle='-', lw=1)

    # --- Beschriftung für WDC mit orthogonalem Versatz ---
    text_pos_on_line = center_rect - 0.5 * wdc_vector
    perp_vec = np.array([wdc_vector[1], -wdc_vector[0]])
    if np.linalg.norm(perp_vec) > 0:
        perp_vec_norm = perp_vec / np.linalg.norm(perp_vec)
    else:
        perp_vec_norm = np.array([0, 0])
    orthogonal_offset = 0.01
    final_text_pos = text_pos_on_line + orthogonal_offset * perp_vec_norm
    angle = np.degrees(np.arctan2(-wdc_vector[1], -wdc_vector[0]))
    ax.text(final_text_pos[0], final_text_pos[1], r'$d_{\mathrm{max}}$',
            color='black', fontsize=20, rotation=angle + 180,
            ha='center', va='center')
    
    # --- Points ---
    ax.plot(true_params[0], true_params[1], 'x', color='red', 
            ms=7, mew=1.2, label=r'True Parameters $\theta$')
    ax.plot(center_rect[0], center_rect[1], '+', color='darkgreen', 
            ms=9, mew=1.5, label=r'Estimate $\hat{\theta}_{rect}$')

    # --- Finalize ---
    ax.set_title('Metrics for Rectangular Region')
    ax.set_xlabel(r'Parameter $a$'); ax.set_ylabel(r'Parameter $b$')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_xlim(-0.12, 0.12); ax.set_ylim(-0.12, 0.12)

    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Rectangle metrics plot saved to: {output_path}")

if __name__ == '__main__':
    create_ellipse_plot(output_path="conceptual_ellipse_metrics.pdf")
    create_rectangle_plot(output_path="conceptual_rectangle_metrics.pdf")