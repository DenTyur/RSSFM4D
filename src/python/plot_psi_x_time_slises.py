import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import time
import os
import gc

basedir = os.path.abspath(os.getcwd())
src_dir = os.path.abspath(os.path.join(basedir, ".."))

x0 = np.load(src_dir + "/arrays_saved/x0.npy")
x1 = np.load(src_dir + "/arrays_saved/x1.npy")
x2 = np.load(src_dir + "/arrays_saved/x2.npy")
x3 = np.load(src_dir + "/arrays_saved/x3.npy")
t = np.load(src_dir + "/arrays_saved/time_evol/t.npy")

index_x_zero = np.zeros(4)
index_x_zero[0] = np.argmin(np.abs(x0))
index_x_zero[1] = np.argmin(np.abs(x1))
index_x_zero[2] = np.argmin(np.abs(x2))
index_x_zero[3] = np.argmin(np.abs(x3))

X = np.meshgrid(x0, x1, x2, x3, indexing="ij")

if not os.path.exists(src_dir + "/imgs/time_evol/psi_x"):
    os.makedirs(src_dir + "/imgs/time_evol/psi_x")

rows, cols = [2, 3]
fig, ax = plt.subplots(rows, cols, figsize=(18, 8), layout="constrained")
ax = ax.flatten()
cb = [None, None, None, None, None, None]

ax_xlabels = ["x1", "x1", "x1", "x2", "x2", "y1"]
ax_ylabels = ["y1", "x2", "y2", "y1", "y2", "y2"]

X1_ind = [0, 0, 0, 2, 2, 1]
X2_ind = [1, 2, 3, 1, 3, 3]

slice_ax_first = [3, 3, 2, 3, 1, 2]
slice_ax_second = [2, 1, 1, 0, 0, 0]


def plot_psi(X, psi, time_step):

    # --- Пояснение, как работают эти срезы (понять сложно, но можно) ---
    # X1_collect = [
    #     X[0][:, :, 0, 0],  # x1 y1
    #     X[0][:, 0, :, 0],  # x1 x2
    #     X[0][:, 0, 0, :],  # x1 y2
    #     X[2][0, :, :, 0],  # x2 y1
    #     X[2][0, 0, :, :],  # x2 y2
    #     X[1][0, :, 0, :],  # y1 y2
    # ]
    #
    # X2_collect = [
    #     X[1][:, :, 0, 0],  # x1 y1
    #     X[2][:, 0, :, 0],  # x1 x2
    #     X[3][:, 0, 0, :],  # x1 y2
    #     X[2][0, :, :, 0],  # x2 y1
    #     X[3][0, 0, :, :],  # x2 y2
    #     X[3][0, :, 0, :],  # y1 y2
    # ]
    # X1_ind = [0,0,0,2,2,1]
    # X2_ind = [1,2,3,1,3,2]
    # axis_ind_first = [3,3,2,3,1,3]
    # axis_ind_second =[2,1,1,0,0,0]
    # ---------------------------------------

    for i in range(cols * rows):
        X1 = (
            X[X1_ind[i]]
            .take(indices=0, axis=slice_ax_first[i])
            .take(indices=0, axis=slice_ax_second[i])
        )

        X2 = (
            X[X2_ind[i]]
            .take(indices=0, axis=slice_ax_first[i])
            .take(indices=0, axis=slice_ax_second[i])
        )

        slice_ind_first = index_x_zero[slice_ax_first[i]]
        slice_ind_second = index_x_zero[slice_ax_second[i]]

        b = ax[i].pcolormesh(
            X1,
            X2,
            np.abs(
                psi.take(indices=slice_ind_first, axis=slice_ax_first[i]).take(
                    indices=slice_ind_second, axis=slice_ax_second[i]
                )
            )
            ** 2,
            cmap=cm.jet,
            shading="auto",
            vmax=1e-5,
            vmin=0,
        )
        cb[i] = plt.colorbar(b, ax=ax[i], fraction=0.046, pad=0.001)
        ax[i].set(xlabel=ax_xlabels[i], ylabel=ax_ylabels[i], aspect="equal")
    fig.savefig(src_dir + f"/imgs/time_evol/psi_x/psi_t_{time_step}.png")
    for i in range(cols * rows):
        ax[i].clear()
        cb[i].remove()
    gc.collect()


for i in range(len(t)):
    ts = time.time()
    psi = np.load(src_dir + f"/arrays_saved/time_evol/psi_x/psi_t_{i}.npy")
    plot_psi(X, psi, i)
    print(f"step {i} of {len(t)}; time of step = {(time.time()-ts):.{5}f}")
