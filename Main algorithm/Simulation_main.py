import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from ractional_Brownian_Field_main import DprwBiFbmSimulator
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator

def fbm_main(sample_size, case = 1, cov_md = 1):
    H_list = np.arange(0, 1, 1/100)[1:]
    if case ==1 :
        # Case 1
        H_list = np.ones(len(H_list))*0.2
        H_list[int(len(H_list)/2):] = 0.8
        # Case 2
    elif case == 2:
        H_list = 0.2 + 0.6*H_list
        # Case 3
    elif case == 3:
        H_list = 0.25 + 2*(H_list - 0.5)**2
    elif case == 4:
        # Case 4
        H_list = 0.5 + 0.3 * np.cos(H_list * np.pi * 6)
    X_ma_ini = DprwBiFbmSimulator(sample_size=sample_size, hurst_parameter=H_list[0], FBM_cov_md=cov_md).get_fbm()
    X_matrix = [X_ma_ini]
    for H in tqdm(H_list[1:]):
        path_H_k = DprwBiFbmSimulator(sample_size=sample_size, hurst_parameter=H, FBM_cov_md=cov_md).get_fbm()
        X_matrix.append(path_H_k)
    X_ma = np.asarray(X_matrix)
    return X_ma, H_list

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

# Draw the random field results under four covariance functions
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

# Test the FBF Hurst estimation
def C_I_boot(fbm_X, B=5000):
    fbm_X = np.array(fbm_X)
    mean_path = np.mean(fbm_X, axis=0)
    sup_dist = []
    for _ in range(B):
        idx = np.random.choice(len(fbm_X), len(fbm_X), replace=True)
        boot_mean = fbm_X[idx].mean(axis=0)
        sup_dist.append(np.max(np.abs(boot_mean - mean_path)))
    sup_dist = np.array(sup_dist)
    c = np.quantile(sup_dist, 0.95)
    lower = mean_path - c
    lower[lower<0] = 0
    upper = mean_path + c
    x = np.linspace(0,1, fbm_X.shape[1])
    return x, lower, upper, mean_path
    
fbm_c1, fbm_c2, fbm_c3, fbm_c4 = [], [], [], []
sample_size = 200
for _ in tqdm(range(100)):
    paths_fbm_c1, H_list_c1 = fbm_main(sample_size, case=1, cov_md = 1)
    paths_fbm_c2, H_list_c2 = fbm_main(sample_size, case=2, cov_md = 1)
    paths_fbm_c3, H_list_c3 = fbm_main(sample_size, case=3, cov_md = 1)
    paths_fbm_c4, H_list_c4 = fbm_main(sample_size, case=4, cov_md = 1)
    fbm_c1.append(paths_fbm_c1)
    fbm_c2.append(paths_fbm_c2)
    fbm_c3.append(paths_fbm_c3)
    fbm_c4.append(paths_fbm_c4)

fbm_H_c1, fbm_H_c2, fbm_H_c3, fbm_H_c4 = [], [], [], []
for k in range(len(fbm_c1)):
    temp1, temp2, temp3, temp4 = [], [], [], []
    for i in tqdm(range(len(paths_fbm_c1))):
        temp1.append(np.mean(QvHurstEstimator(mbm_series=fbm_c1[k][i], alpha=0.2).holder_exponents))
        temp2.append(np.mean(QvHurstEstimator(mbm_series=fbm_c2[k][i], alpha=0.2).holder_exponents))
        temp3.append(np.mean(QvHurstEstimator(mbm_series=fbm_c3[k][i], alpha=0.2).holder_exponents))
        temp4.append(np.mean(QvHurstEstimator(mbm_series=fbm_c4[k][i], alpha=0.2).holder_exponents))
    fbm_H_c1.append(temp1)
    fbm_H_c2.append(temp2)
    fbm_H_c3.append(temp3)
    fbm_H_c4.append(temp4)

c1_plt = C_I_boot(fbm_H_c1)
c2_plt = C_I_boot(fbm_H_c2)
c3_plt = C_I_boot(fbm_H_c3)
c4_plt = C_I_boot(fbm_H_c4)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
axs[0, 0].plot(c1_plt[0], c1_plt[3])
axs[0, 0].fill_between(c1_plt[0], c1_plt[1], c1_plt[2], alpha=0.3)
axs[0, 0].grid()
axs[0, 0].set_xticklabels([])
axs[0, 0].set_title('Scenario 1', fontsize=13)
axs[0, 0].set_ylabel('H', fontsize=13)
axs[0, 0].set_ylim(0.1,0.9)

axs[0, 1].plot(c2_plt[0], c2_plt[3])
axs[0, 1].fill_between(c2_plt[0], c2_plt[1], c2_plt[2], alpha=0.3)
axs[0, 1].grid()
axs[0, 1].set_xticklabels([])
axs[0, 1].set_yticklabels([])
axs[0, 1].set_title('Scenario 2', fontsize=13)
axs[0, 1].set_ylim(0.1,0.9)

axs[1, 0].plot(c3_plt[0], c3_plt[3])
axs[1, 0].fill_between(c3_plt[0], c3_plt[1], c3_plt[2], alpha=0.3)
axs[1, 0].grid()
axs[1, 0].set_title('Scenario 3', fontsize=13)
axs[1, 0].set_ylabel('H', fontsize=13)
axs[1, 0].set_xlabel('k/M', fontsize=13)
axs[1, 0].set_ylim(0.1,0.9)

axs[1, 1].plot(c4_plt[0], c4_plt[3])
axs[1, 1].fill_between(c4_plt[0], c4_plt[1], c4_plt[2], alpha=0.3)
axs[1, 1].set_yticklabels([])
axs[1, 1].grid()
axs[1, 1].set_title('Scenario 4', fontsize=13)
axs[1, 1].set_xlabel('k/M', fontsize=13)
axs[1, 1].set_ylim(0.1,0.9)

plt.savefig('FBM_H_est.jpg', dpi=500)
plt.show()

# Test Hurst under different path length
fbm_c256, fbm_c512, fbm_c1024 = [], [], []
for _ in tqdm(range(100)):
    sample_size = 256
    paths_fbm_c256, H_list_c1 = fbm_main(sample_size, case=1, cov_md = 1)
    sample_size = 512
    paths_fbm_c512, H_list_c2 = fbm_main(sample_size, case=1, cov_md = 1)
    sample_size = 1024
    paths_fbm_c1024, H_list_c3 = fbm_main(sample_size, case=1, cov_md = 1)
    fbm_c256.append(paths_fbm_c256)
    fbm_c512.append(paths_fbm_c512)
    fbm_c1024.append(paths_fbm_c1024)

fbm_H_c256_leng, fbm_H_c512_leng, fbm_H_c1024_leng = [], [], []
for k in range(len(fbm_c256)):
    temp1, temp2, temp3, temp4 = [], [], [], []
    for i in tqdm(range(len(fbm_c256[0]))):
        temp1.append(np.mean(QvHurstEstimator(mbm_series=fbm_c256[k][i], alpha=0.2).holder_exponents))
        temp2.append(np.mean(QvHurstEstimator(mbm_series=fbm_c512[k][i], alpha=0.2).holder_exponents))
        temp3.append(np.mean(QvHurstEstimator(mbm_series=fbm_c1024[k][i], alpha=0.2).holder_exponents))
    fbm_H_c256_leng.append(temp1)
    fbm_H_c512_leng.append(temp2)
    fbm_H_c1024_leng.append(temp3)
    
fbm_H_c256_leng = np.array(fbm_H_c256_leng)
fbm_H_c512_leng = np.array(fbm_H_c512_leng)
fbm_H_c1024_leng = np.array(fbm_H_c1024_leng)

target_H = np.ones(shape=fbm_H_c1024_leng.shape[1])
target_H[:49] = 0.2
target_H[49:] = 0.8

def H_rmse(test, target):
    return np.sqrt(np.sum((test - target)**2)/len(test))

rmse_256, rmse_512, rmse_1024 = [], [], []
for i in range(len(fbm_H_c256_leng)):
    rmse_256.append(H_rmse(fbm_H_c256_leng[i], target_H))
    rmse_512.append(H_rmse(fbm_H_c512_leng[i], target_H))
    rmse_1024.append(H_rmse(fbm_H_c1024_leng[i], target_H))
print(np.mean(rmse_256))
print(np.mean(rmse_512))
print(np.mean(rmse_1024))

rmse_length_all = np.array([rmse_256, rmse_512, rmse_1024])
rmse_length_all = np.transpose(rmse_length_all)

fig, ax = plt.subplots()

bplot = ax.boxplot(rmse_length_all,
                   patch_artist=True,
                   tick_labels=[256, 512, 1024])
plt.grid()
ax.set_xlabel('Length', fontsize=13)
ax.set_ylabel('Root mean square error', fontsize=13)
plt.savefig('H_est_fbm_c1_rmse.jpg', dpi=500)

