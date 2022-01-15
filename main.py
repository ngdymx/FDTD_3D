import FDTD
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

if __name__ == '__main__':
    dx = 0.1
    dy = 0.1
    dz = 0.1
    Steps = 200
    c = 3e8
    mu = 4 * np.pi * 1e-7
    ep = 1 / mu / c / c
    dt = 0.8 * 1 / np.sqrt(1 / (dx ** 2) + 1 / (dy ** 2) + 1 / (dz ** 2)) / 3e8
    space = FDTD.FDTD_3D_space(25, 25, 25, dt, dx, dy, dz, ep=ep, mu=mu)
    space.set_pml('B', 'L', 'B', 10)
    space.set_pml('F', 'R', 'T', 10)
    # space.set_tfsf_boundary([10, 30], [40, 170])
    # space.add_material(ep * 8, mu, 0, 10)

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # X = np.arange(25)
    # Y = np.arange(25)
    # X, Y = np.meshgrid(X, Y)
    figure = plt.figure(figsize=(10, 10), dpi=100)

    for t in range(Steps):
        plt.clf()
        space.update()
        space.apply_src([12, 12, 12], 10*np.sin(2 * np.pi * t / 50), stype='E')
        Ez = space.export_value_Ez()
        plt.imshow(Ez[:, :, 12], vmin=-0.01, vmax=0.01)
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # surf = ax.plot_surface(X, Y, Ez[:, :, 10], rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, linewidth=0, antialiased=False)
        # ax.set_zlim(-1, 1)
        # fig.colorbar(surf, shrink=0.5, aspect=10)

        # ax = fig.add_subplot(1, 2, 2, projection='3d')
        # X, Y, Ez[:, :, 10] = get_test_data(0.24)
        # ax.plot_wireframe(X, Y, Ez[:, :, 10], rstride=10, cstride=10)
        plt.pause(.01)
