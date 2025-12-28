import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from ractional_Brownian_Field_main import fbm_main


# Visualization
def plot_fbm(paths, t_list, H_k_list, H_list, save_figure, fig_title = 'Random field value of FBM'):
    K, T = np.meshgrid(t_list, H_k_list)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2.5, 1, 1], height_ratios=[1, 1], wspace=0.21, hspace=0.16)

    ax_big = fig.add_subplot(gs[:, 0], projection='3d')
    surf = ax_big.plot_surface(K, T, paths, cmap='viridis', edgecolor='none')
    
    ax_big.set_xlabel("Time t", fontsize=13)
    ax_big.set_ylabel("k/M", fontsize=13)
    ax_big.set_zlim(np.min(paths), np.max(paths))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax_big.zaxis.set_major_formatter(formatter)
    ax_big.figure.canvas.draw()
    offset_text = ax_big.zaxis.get_offset_text().get_text()
    ax_big.zaxis.get_offset_text().set_visible(False)

    ax_big.text2D(
        0.95, 0.8, offset_text,
        transform=ax_big.transAxes,
        ha='left', va='center'
    )
    ax_big.grid(False)

    x_line = np.linspace(0, 1, 100)
    y_line = np.zeros_like(x_line)
    z_line = np.ones_like(x_line)*np.max(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    x_line = np.linspace(0, 1, 100)
    y_line = np.ones_like(x_line)
    z_line = np.ones_like(x_line)*np.max(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    x_line = np.linspace(0, 1, 100)
    y_line = np.zeros_like(x_line)
    z_line = np.ones_like(x_line)*np.min(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1)

    x_line = np.linspace(0, 1, 100)
    y_line = np.ones_like(x_line)
    z_line = np.ones_like(x_line)*np.min(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1)

    y_line = np.linspace(0, 1, 100)
    x_line = np.zeros_like(y_line)
    z_line = np.ones_like(y_line)*np.max(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    y_line = np.linspace(0, 1, 100)
    x_line = np.ones_like(y_line)
    z_line = np.ones_like(y_line)*np.max(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    y_line = np.linspace(0, 1, 100)
    x_line = np.ones_like(y_line)
    z_line = np.ones_like(y_line)*np.min(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    y_line = np.linspace(0, 1, 100)
    x_line = np.zeros_like(y_line)
    z_line = np.ones_like(y_line)*np.min(paths)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1)

    z_line = np.linspace(np.min(paths), np.max(paths), 100)
    x_line = np.ones_like(z_line)
    y_line = np.ones_like(z_line)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    z_line = np.linspace(np.min(paths), np.max(paths), 100)
    x_line = np.zeros_like(z_line)
    y_line = np.zeros_like(z_line)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    z_line = np.linspace(np.min(paths), np.max(paths), 100)
    x_line = np.zeros_like(z_line)
    y_line = np.ones_like(z_line)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1)

    z_line = np.linspace(np.min(paths), np.max(paths), 100)
    x_line = np.ones_like(z_line)
    y_line = np.zeros_like(z_line)
    ax_big.plot(x_line, y_line, z_line, color='red', linestyle='--', linewidth=1, zorder=10)

    norm = surf.norm
    cmap = surf.cmap
    pos_ind = np.arange(20, 81, 20)

    for i in range(4):
        row = i // 2
        col = i % 2 + 1 
        ax = fig.add_subplot(gs[row, col])
        spec_line = paths[pos_ind[i]]
        points = np.array([t_list, spec_line]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
        lc.set_array(spec_line)
        ax.add_collection(lc)
        temp_size = np.max(spec_line) - np.min(spec_line)
        ax.set_ylim(np.min(spec_line)-0.1*temp_size, np.max(spec_line)+0.1*temp_size)

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
        offset = ax.yaxis.get_offset_text()
        offset.set_x(-0.1)
        offset.set_y(1.0) 

        sub_fig_title = 'H(' + str(pos_ind[i]/100) + ')=' + f"{H_list[pos_ind[i]]:.{3}f}".rstrip('0').rstrip('.')
        ax.set_title(sub_fig_title, fontsize=14)
        ax.grid()
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time t', fontsize=13)
    # fig.suptitle(fig_title, fontsize=20, x = 0.54, y=0.98)
    fig.colorbar(surf, ax=fig.get_axes(), orientation='vertical', fraction=0.02, pad=0.02)
    # plt.savefig(save_figure, dpi=500)
    plt.show()

sample_size = 400
H_k_list = np.arange(0, 1, 1/100)[1:]
t_list = np.linspace(0,1,sample_size)

paths_fbm_c1, H_list_c1 = fbm_main(sample_size, case=1, cov_md = 1)
paths_fbm_c2, H_list_c2 = fbm_main(sample_size, case=2, cov_md = 1)
paths_fbm_c3, H_list_c3 = fbm_main(sample_size, case=3, cov_md = 1)
paths_fbm_c4, H_list_c4 = fbm_main(sample_size, case=4, cov_md = 1)
plot_fbm(paths_fbm_c1, t_list, H_k_list, H_list_c1, 'paths_fbm_c1.jpg', fig_title='Random field of FBF (Scenario 1)')
plot_fbm(paths_fbm_c2, t_list, H_k_list, H_list_c2, 'paths_fbm_c2.jpg', fig_title='Random field of FBF (Scenario 2)')
plot_fbm(paths_fbm_c3, t_list, H_k_list, H_list_c3, 'paths_fbm_c3.jpg', fig_title='Random field of FBF (Scenario 3)')
plot_fbm(paths_fbm_c4, t_list, H_k_list, H_list_c4, 'paths_fbm_c4.jpg', fig_title='Random field of FBF (Scenario 4)')

paths_sub_fbm_cov_c1, H_list_c1 = fbm_main(sample_size, case=1, cov_md = 2)
paths_sub_fbm_cov_c2, H_list_c2 = fbm_main(sample_size, case=2, cov_md = 2)
paths_sub_fbm_cov_c3, H_list_c3 = fbm_main(sample_size, case=3, cov_md = 2)
paths_sub_fbm_cov_c4, H_list_c4 = fbm_main(sample_size, case=4, cov_md = 2)
plot_fbm(paths_sub_fbm_cov_c1, t_list, H_k_list, H_list_c1, 'paths_sub_fbm_cov_c1.jpg', fig_title='Random field of Sub-FBF (Scenario 1)')
plot_fbm(paths_sub_fbm_cov_c2, t_list, H_k_list, H_list_c2, 'paths_sub_fbm_cov_c2.jpg', fig_title='Random field of Sub-FBF (Scenario 2)')
plot_fbm(paths_sub_fbm_cov_c3, t_list, H_k_list, H_list_c3, 'paths_sub_fbm_cov_c3.jpg', fig_title='Random field of Sub-FBF (Scenario 3)')
plot_fbm(paths_sub_fbm_cov_c4, t_list, H_k_list, H_list_c4, 'paths_sub_fbm_cov_c4.jpg', fig_title='Random field of Sub-FBF (Scenario 4)')

paths_bi_fbm_cov_c1, H_list_c1 = fbm_main(sample_size, case=1, cov_md = 3)
paths_bi_fbm_cov_c2, H_list_c2 = fbm_main(sample_size, case=2, cov_md = 3)
paths_bi_fbm_cov_c3, H_list_c3 = fbm_main(sample_size, case=3, cov_md = 3)
paths_bi_fbm_cov_c4, H_list_c4 = fbm_main(sample_size, case=4, cov_md = 3)
plot_fbm(paths_bi_fbm_cov_c1, t_list, H_k_list, H_list_c1, 'paths_bi_fbm_cov_c1.jpg', fig_title='Random field of Bi-FBF (Scenario 1)')
plot_fbm(paths_bi_fbm_cov_c2, t_list, H_k_list, H_list_c2, 'paths_bi_fbm_cov_c2.jpg', fig_title='Random field of Bi-FBF (Scenario 2)')
plot_fbm(paths_bi_fbm_cov_c3, t_list, H_k_list, H_list_c3, 'paths_bi_fbm_cov_c3.jpg', fig_title='Random field of Bi-FBF (Scenario 3)')
plot_fbm(paths_bi_fbm_cov_c4, t_list, H_k_list, H_list_c4, 'paths_bi_fbm_cov_c4.jpg', fig_title='Random field of Bi-FBF (Scenario 4)')

paths_tri_fbm_c1, H_list_c1 = fbm_main(sample_size, case=1, cov_md = 4)
paths_tri_fbm_c2, H_list_c2 = fbm_main(sample_size, case=2, cov_md = 4)
paths_tri_fbm_c3, H_list_c3 = fbm_main(sample_size, case=3, cov_md = 4)
paths_tri_fbm_c4, H_list_c4 = fbm_main(sample_size, case=4, cov_md = 4)
plot_fbm(paths_tri_fbm_c1, t_list, H_k_list, H_list_c1, 'paths_tri_fbm_c1.jpg', fig_title='Random field of Tri-FBF (Scenario 1)')
plot_fbm(paths_tri_fbm_c2, t_list, H_k_list, H_list_c2, 'paths_tri_fbm_c2.jpg', fig_title='Random field of Tri-FBF (Scenario 2)')
plot_fbm(paths_tri_fbm_c3, t_list, H_k_list, H_list_c3, 'paths_tri_fbm_c3.jpg', fig_title='Random field of Tri-FBF (Scenario 3)')
plot_fbm(paths_tri_fbm_c4, t_list, H_k_list, H_list_c4, 'paths_tri_fbm_c4.jpg', fig_title='Random field of Tri-FBF (Scenario 4)')

