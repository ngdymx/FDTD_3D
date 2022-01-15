import numpy as np


class fdtd_3d_systolic_block:
    def __init__(self):
        self.Bz = 0
        self.Bz_last = 0
        self.Bzx = 0
        self.Bzy = 0
        self.Bzx_last = 0
        self.Bzy_last = 0
        self.Hz = 0
        self.Bx = 0
        self.Bx_last = 0
        self.Bxy = 0
        self.Bxz = 0
        self.Bxy_last = 0
        self.Bxz_last = 0
        self.Hx = 0
        self.By = 0
        self.By_last = 0
        self.Byx = 0
        self.Byz = 0
        self.Byx_last = 0
        self.Byz_last = 0
        self.Hy = 0

        self.Dz = 0
        self.Dz_last = 0
        self.Dzx = 0
        self.Dzy = 0
        self.Dzx_last = 0
        self.Dzy_last = 0
        self.Ez = 0
        self.Dx = 0
        self.Dx_last = 0
        self.Dxy = 0
        self.Dxz = 0
        self.Dxy_last = 0
        self.Dxz_last = 0
        self.Ex = 0
        self.Dy = 0
        self.Dy_last = 0
        self.Dyx = 0
        self.Dyz = 0
        self.Dyx_last = 0
        self.Dyz_last = 0
        self.Ey = 0

        self.dt = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.ep = 0
        self.mu = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.sigma_x = 0
        self.sigma_y = 0
        self.sigma_z = 0
        self.kdx1 = 0
        self.kdx2 = 0
        self.kdx3 = 0
        self.kex1 = 0
        self.kex2 = 0
        self.kex3 = 0
        self.kdy1 = 0
        self.kdy2 = 0
        self.kdy3 = 0
        self.key1 = 0
        self.key2 = 0
        self.key3 = 0
        self.kdz1 = 0
        self.kdz2 = 0
        self.kdz3 = 0
        self.kez1 = 0
        self.kez2 = 0
        self.kez3 = 0
        self.kbx1 = 0
        self.kbx2 = 0
        self.kbx3 = 0
        self.khx1 = 0
        self.khx2 = 0
        self.khx3 = 0
        self.kby1 = 0
        self.kby2 = 0
        self.kby3 = 0
        self.khy1 = 0
        self.khy2 = 0
        self.khy3 = 0
        self.kbz1 = 0
        self.kbz2 = 0
        self.kbz3 = 0
        self.khz1 = 0
        self.khz2 = 0
        self.khz3 = 0

    def set_block(self, dt, dx, dy, dz, ep, mu, kx=1, ky=1, kz=1, sigma_x=0, sigma_y=0, sigma_z=0):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ep = ep
        self.mu = mu
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.kdx1 = (2 * ep * ky - sigma_y * dt) / (2 * ep * ky + sigma_y * dt)
        self.kdx2 = 2 * ep * dt / (2 * ep * ky + sigma_y * dt) / dy
        self.kdx3 = 2 * ep * dt / (2 * ep * ky + sigma_y * dt) / dz
        self.kex1 = (2 * ep * kz - sigma_z * dt) / (2 * ep * kz + sigma_z * dt)
        self.kex2 = (2 * ep * kx + sigma_x * dt) / (2 * ep * kz + sigma_z * dt) / ep
        self.kex3 = (2 * ep * kx - sigma_x * dt) / (2 * ep * kz + sigma_z * dt) / ep

        self.kdy1 = (2 * ep * kz - sigma_z * dt) / (2 * ep * kz + sigma_z * dt)
        self.kdy2 = 2 * ep * dt / (2 * ep * kz + sigma_z * dt) / dz
        self.kdy3 = 2 * ep * dt / (2 * ep * kz + sigma_z * dt) / dx
        self.key1 = (2 * ep * kx - sigma_x * dt) / (2 * ep * kx + sigma_x * dt)
        self.key2 = (2 * ep * ky + sigma_y * dt) / (2 * ep * kx + sigma_x * dt) / ep
        self.key3 = (2 * ep * ky - sigma_y * dt) / (2 * ep * kx + sigma_x * dt) / ep

        self.kdz1 = (2 * ep * kx - sigma_x * dt) / (2 * ep * kx + sigma_x * dt)
        self.kdz2 = 2 * ep * dt / (2 * ep * kx + sigma_x * dt) / dx
        self.kdz3 = 2 * ep * dt / (2 * ep * kx + sigma_x * dt) / dy
        self.kez1 = (2 * ep * ky - sigma_y * dt) / (2 * ep * ky + sigma_y * dt)
        self.kez2 = (2 * ep * kz + sigma_z * dt) / (2 * ep * ky + sigma_y * dt) / ep
        self.kez3 = (2 * ep * kz - sigma_z * dt) / (2 * ep * ky + sigma_y * dt) / ep

        self.kbx1 = (2 * ep * ky - sigma_y * dt) / (2 * ep * ky + sigma_y * dt)
        self.kbx2 = 2 * ep * dt / (2 * ep * ky + sigma_y * dt) / dy
        self.kbx3 = 2 * ep * dt / (2 * ep * ky + sigma_y * dt) / dz
        self.khx1 = (2 * ep * kz - sigma_z * dt) / (2 * ep * kz + sigma_z * dt)
        self.khx2 = (2 * ep * kx + sigma_x * dt) / (2 * ep * kz + sigma_z * dt) / mu
        self.khx3 = (2 * ep * kx - sigma_x * dt) / (2 * ep * kz + sigma_z * dt) / mu

        self.kby1 = (2 * ep * kz - sigma_z * dt) / (2 * ep * kz + sigma_z * dt)
        self.kby2 = 2 * ep * dt / (2 * ep * kz + sigma_z * dt) / dz
        self.kby3 = 2 * ep * dt / (2 * ep * kz + sigma_z * dt) / dx
        self.khy1 = (2 * ep * kx - sigma_x * dt) / (2 * ep * kx + sigma_x * dt)
        self.khy2 = (2 * ep * ky + sigma_y * dt) / (2 * ep * kx + sigma_x * dt) / mu
        self.khy3 = (2 * ep * ky - sigma_y * dt) / (2 * ep * kx + sigma_x * dt) / mu

        self.kbz1 = (2 * ep * kx - sigma_x * dt) / (2 * ep * kx + sigma_x * dt)
        self.kbz2 = 2 * ep * dt / (2 * ep * kx + sigma_x * dt) / dx
        self.kbz3 = 2 * ep * dt / (2 * ep * kx + sigma_x * dt) / dy
        self.khz1 = (2 * ep * ky - sigma_y * dt) / (2 * ep * ky + sigma_y * dt)
        self.khz2 = (2 * ep * kz + sigma_z * dt) / (2 * ep * ky + sigma_y * dt) / mu
        self.khz3 = (2 * ep * kz - sigma_z * dt) / (2 * ep * ky + sigma_y * dt) / mu

    def update_parameters(self):
        self.kdx1 = (2 * self.ep * self.ky - self.sigma_y * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt)
        self.kdx2 = 2 * self.ep * self.dt / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.dy
        self.kdx3 = 2 * self.ep * self.dt / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.dz
        self.kex1 = (2 * self.ep * self.kz - self.sigma_z * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt)
        self.kex2 = (2 * self.ep * self.kx + self.sigma_x * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.ep
        self.kex3 = (2 * self.ep * self.kx - self.sigma_x * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.ep

        self.kdy1 = (2 * self.ep * self.kz - self.sigma_z * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt)
        self.kdy2 = 2 * self.ep * self.dt / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.dz
        self.kdy3 = 2 * self.ep * self.dt / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.dx
        self.key1 = (2 * self.ep * self.kx - self.sigma_x * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt)
        self.key2 = (2 * self.ep * self.ky + self.sigma_y * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.ep
        self.key3 = (2 * self.ep * self.ky - self.sigma_y * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.ep

        self.kdz1 = (2 * self.ep * self.kx - self.sigma_x * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt)
        self.kdz2 = 2 * self.ep * self.dt / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.dx
        self.kdz3 = 2 * self.ep * self.dt / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.dy
        self.kez1 = (2 * self.ep * self.ky - self.sigma_y * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt)
        self.kez2 = (2 * self.ep * self.kz + self.sigma_z * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.ep
        self.kez3 = (2 * self.ep * self.kz - self.sigma_z * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.ep

        self.kbx1 = (2 * self.ep * self.ky - self.sigma_y * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt)
        self.kbx2 = 2 * self.ep * self.dt / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.dy
        self.kbx3 = 2 * self.ep * self.dt / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.dz
        self.khx1 = (2 * self.ep * self.kz - self.sigma_z * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt)
        self.khx2 = (2 * self.ep * self.kx + self.sigma_x * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.mu
        self.khx3 = (2 * self.ep * self.kx - self.sigma_x * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.mu

        self.kby1 = (2 * self.ep * self.kz - self.sigma_z * self.dt) / (2 * self.ep * self.kz + self.sigma_z * self.dt)
        self.kby2 = 2 * self.ep * self.dt / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.dz
        self.kby3 = 2 * self.ep * self.dt / (2 * self.ep * self.kz + self.sigma_z * self.dt) / self.dx
        self.khy1 = (2 * self.ep * self.kx - self.sigma_x * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt)
        self.khy2 = (2 * self.ep * self.ky + self.sigma_y * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.mu
        self.khy3 = (2 * self.ep * self.ky - self.sigma_y * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.mu

        self.kbz1 = (2 * self.ep * self.kx - self.sigma_x * self.dt) / (2 * self.ep * self.kx + self.sigma_x * self.dt)
        self.kbz2 = 2 * self.ep * self.dt / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.dx
        self.kbz3 = 2 * self.ep * self.dt / (2 * self.ep * self.kx + self.sigma_x * self.dt) / self.dy
        self.khz1 = (2 * self.ep * self.ky - self.sigma_y * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt)
        self.khz2 = (2 * self.ep * self.kz + self.sigma_z * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.mu
        self.khz3 = (2 * self.ep * self.kz - self.sigma_z * self.dt) / (2 * self.ep * self.ky + self.sigma_y * self.dt) / self.mu

    def apply_src(self, value, stype='E'):
        if stype == 'E':
            self.Ez = self.Ez + value
        else:
            self.Hz = self.Hz + value

    def update_Bxy(self, Ez):
        self.Bxy = self.kbx1 * self.Bxy_last - self.kbx2 * (Ez - self.Ez)

    def update_Bxz(self, Ey):
        self.Bxz = self.kbx1 * self.Bxz_last + self.kbx3 * (Ey - self.Ey)

    def update_Bx(self):
        self.Bx = self.Bxz + self.Bxy

    def update_Hx(self):
        self.Hx = self.khx1 * self.Hx + self.khx2 * self.Bx - self.khx3 * self.Bx_last

    def update_Bxz_last(self):
        self.Bxz_last = self.Bxz

    def update_Bxy_last(self):
        self.Bxy_last = self.Bxy

    def update_Bx_last(self):
        self.Bx_last = self.Bx

    def update_Byz(self, Ex):
        self.Byz = self.kby1 * self.Byz_last - self.kby2 * (Ex - self.Ex)

    def update_Byx(self, Ez):
        self.Byx = self.kby1 * self.Byx_last + self.kby3 * (Ez - self.Ez)

    def update_By(self):
        self.By = self.Byz + self.Byx

    def update_Hy(self):
        self.Hy = self.khy1 * self.Hy + self.khy2 * self.By - self.khy3 * self.By_last

    def update_Byz_last(self):
        self.Byz_last = self.Byz

    def update_Byx_last(self):
        self.Byx_last = self.Byx

    def update_By_last(self):
        self.By_last = self.By

    def update_Bzx(self, Ey):
        self.Bzx = self.kbz1 * self.Bzx_last - self.kbz2 * (Ey - self.Ey)

    def update_Bzy(self, Ex):
        self.Bzy = self.kbz1 * self.Bzy_last + self.kby3 * (Ex - self.Ex)

    def update_Bz(self):
        self.Bz = self.Bzx + self.Bzy

    def update_Hz(self):
        self.Hz = self.khz1 * self.Hz + self.khz2 * self.Bz - self.khz3 * self.Bz_last

    def update_Bzx_last(self):
        self.Bzx_last = self.Bzx

    def update_Bzy_last(self):
        self.Bzy_last = self.Bzy

    def update_Bz_last(self):
        self.Bz_last = self.Bz

    def update_Dxy(self, Hz):
        self.Dxy = self.kdx1 * self.Dxy_last + self.kdx2 * (self.Hz - Hz)

    def update_Dxz(self, Hy):
        self.Dxz = self.kdx1 * self.Dxz_last - self.kdx3 * (self.Hy - Hy)

    def update_Dx(self):
        self.Dx = self.Dxy + self.Dxz

    def update_Ex(self):
        self.Ex = self.kex1 * self.Ex + self.kex2 * self.Dx - self.kex3 * self.Dx_last

    def update_Dxy_last(self):
        self.Dxy_last = self.Dxy

    def update_Dxz_last(self):
        self.Dxz_last = self.Dxz

    def update_Dx_last(self):
        self.Dx_last = self.Dx

    def update_Dyz(self, Hx):
        self.Dyz = self.kdy1 * self.Dyz_last + self.kdy2 * (self.Hx - Hx)

    def update_Dyx(self, Hz):
        self.Dyx = self.kdy1 * self.Dyx_last - self.kdy3 * (self.Hz - Hz)

    def update_Dy(self):
        self.Dy = self.Dyx + self.Dyz

    def update_Ey(self):
        self.Ey = self.key1 * self.Ey + self.key2 * self.Dy - self.key3 * self.Dy_last

    def update_Dyx_last(self):
        self.Dyx_last = self.Dyx

    def update_Dyz_last(self):
        self.Dyz_last = self.Dyz

    def update_Dy_last(self):
        self.Dy_last = self.Dy

    def update_Dzx(self, Hy):
        self.Dzx = self.kdz1 * self.Dzx_last + self.kdz2 * (self.Hy - Hy)

    def update_Dzy(self, Hx):
        self.Dzy = self.kdz1 * self.Dzy_last - self.kdz3 * (self.Hx - Hx)

    def update_Dz(self):
        self.Dz = self.Dzy + self.Dzx

    def update_Ez(self):
        self.Ez = self.kez1 * self.Ez + self.kez2 * self.Dz - self.kez3 * self.Dz_last

    def update_Dzy_last(self):
        self.Dzy_last = self.Dzy

    def update_Dzx_last(self):
        self.Dzx_last = self.Dzx

    def update_Dz_last(self):
        self.Dz_last = self.Dz


class FDTD_3D_space:
    def __init__(self, x_nodes, y_nodes, z_nodes, dt, dx, dy, dz, mu, ep):
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.z_nodes = z_nodes
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.mu = mu
        self.ep = ep
        self.systolic_blocks1 = []
        for i in range(x_nodes):
            surf = []
            for j in range(y_nodes):
                col = []
                for k in range(z_nodes):
                    col.append(fdtd_3d_systolic_block())
                    col[k].set_block(dt, dx, dy, dz, ep=ep, mu=mu, sigma_x=0, sigma_y=0, sigma_z=0)
                surf.append(col.copy())
                del col
            self.systolic_blocks1.append(surf.copy())
            del surf

        self.Hz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hzx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hzy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bzx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bzy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bz_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bzx_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bzy_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hxz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hxy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bxz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bxy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bx_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bxz_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Bxy_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hyx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hyz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.By = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Byx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Byz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.By_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Byx_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Byz_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ex = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Exy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Exz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dxy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dxz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dx_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dxy_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dxz_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ey = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Eyx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Eyz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dyx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dyz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dy_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dyx_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dyz_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ez = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ezy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ezx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dzy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dzx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dz_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dzy_last = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Dzx_last = np.zeros([x_nodes, y_nodes, z_nodes])

    def apply_src(self, pos, value, stype='E'):
        x = pos[0]
        y = pos[1]
        z = pos[2]
        self.systolic_blocks1[x][y][z].apply_src(value, stype)

    def update(self):
        # update 3D Bxz
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    if k == self.z_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Bxz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Bxz(self.systolic_blocks1[i][j][k + 1].Ey)

        # update 3D Bxy
        for i in range(self.x_nodes):
            for k in range(self.z_nodes):
                for j in range(self.y_nodes):
                    if j == self.y_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Bxy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Bxy(self.systolic_blocks1[i][j + 1][k].Ez)

        # update 3D Bx
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Bx()

        # update 3D Hx
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Hx()

        # update 3D Byx
        for j in range(self.y_nodes):
            for k in range(self.z_nodes):
                for i in range(self.x_nodes):
                    if i == self.x_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Byx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Byx(self.systolic_blocks1[i + 1][j][k].Ez)

        # update 3D Byz
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    if k == self.z_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Byz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Byz(self.systolic_blocks1[i][j][k + 1].Ex)

        # update 3D By
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_By()

        # update 3D Hy
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Hy()

        # update 3D Bzy
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    if j == self.y_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Bzy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Bzy(self.systolic_blocks1[i][j + 1][k].Ex)

        # update 3D Bzx
        for k in range(self.z_nodes):
            for j in range(self.y_nodes):
                for i in range(self.x_nodes):
                    if i == self.x_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Bzx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Bzx(self.systolic_blocks1[i + 1][j][k].Ey)

        # update 3D Bz
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[i][j][k].update_Bz()

        # update 3D Hz
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[i][j][k].update_Hz()

        # update 3D Dxy
        for i in range(self.x_nodes):
            for k in range(self.z_nodes):
                for j in range(self.y_nodes):
                    if j == 0:
                        self.systolic_blocks1[i][j][k].update_Dxy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Dxy(self.systolic_blocks1[i][j - 1][k].Hz)

        # update 3D Dxz
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    if k == 0:
                        self.systolic_blocks1[i][j][k].update_Dxz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Dxz(self.systolic_blocks1[i][j][k - 1].Hy)

        # update 3D Dx
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Dx()

        # update 3D Ex
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Ex()

        # update 3D Dyz
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    if k == 0:
                        self.systolic_blocks1[i][j][k].update_Dyz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Dyz(self.systolic_blocks1[i][j][k - 1].Hx)

        # update 3D Dyx
        for j in range(self.y_nodes):
            for k in range(self.z_nodes):
                for i in range(self.x_nodes):
                    if i == 0:
                        self.systolic_blocks1[i][j][k].update_Dyx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Dyx(self.systolic_blocks1[i - 1][j][k].Hz)

        # update 3D Dy
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Dy()

        # update 3D Ey
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Ey()

        # update 3D Dzx
        for k in range(self.z_nodes):
            for j in range(self.y_nodes):
                for i in range(self.x_nodes):
                    if i == 0:
                        self.systolic_blocks1[i][j][k].update_Dzx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Dzx(self.systolic_blocks1[i - 1][j][k].Hy)

        # update 3D Dzy
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    if j == 0:
                        self.systolic_blocks1[i][j][k].update_Dzy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Dzy(self.systolic_blocks1[i][j - 1][k].Hx)

        # update 3D Dz
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[i][j][k].update_Dz()

        # update 3D Ez
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[i][j][k].update_Ez()

        # update 3D Bz_last, Bzx_last, Bzy_last, Bx_last, Bxz_last, Bxy_last, Byx_last, Byz_last,
        # Dz_last, Dzx_last, Dzy_last, Dx_last, Dxz_last, Dxy_last, Dyx_last, Dyz_last
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Bxz_last()
                    self.systolic_blocks1[i][j][k].update_Bxy_last()
                    self.systolic_blocks1[i][j][k].update_Bx_last()
                    self.systolic_blocks1[i][j][k].update_Byx_last()
                    self.systolic_blocks1[i][j][k].update_Byz_last()
                    self.systolic_blocks1[i][j][k].update_By_last()
                    self.systolic_blocks1[i][j][k].update_Bzx_last()
                    self.systolic_blocks1[i][j][k].update_Bzy_last()
                    self.systolic_blocks1[i][j][k].update_Bz_last()
                    self.systolic_blocks1[i][j][k].update_Dxz_last()
                    self.systolic_blocks1[i][j][k].update_Dxy_last()
                    self.systolic_blocks1[i][j][k].update_Dx_last()
                    self.systolic_blocks1[i][j][k].update_Dyz_last()
                    self.systolic_blocks1[i][j][k].update_Dyx_last()
                    self.systolic_blocks1[i][j][k].update_Dy_last()
                    self.systolic_blocks1[i][j][k].update_Dzx_last()
                    self.systolic_blocks1[i][j][k].update_Dzy_last()
                    self.systolic_blocks1[i][j][k].update_Dz_last()

    def set_pml(self, x_side, y_side, z_side,  d, R0=1e-16, M=3, ep=8.85e-12):
        sigmax_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dx
        sigmay_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dy
        sigmaz_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dz
        Pfront = np.power((np.arange(d) / d), M) * sigmax_max
        Pright = np.power((np.arange(d) / d), M) * sigmay_max
        Ptop = np.power((np.arange(d) / d), M) * sigmaz_max
        if x_side == 'B':
            for i in range(d):
                for j in range(self.y_nodes):
                    for k in range(self.z_nodes):
                        self.systolic_blocks1[i][j][k].sigma_x = Pfront[d - 1 - i]
                        self.systolic_blocks1[i][j][k].sigma_y = Pfront[d - 1 - i]
                        self.systolic_blocks1[i][j][k].sigma_z = Pfront[d - 1 - i]
        else:
            for i in range(d):
                for j in range(self.y_nodes):
                    for k in range(self.z_nodes):
                        self.systolic_blocks1[self.x_nodes - d + i][j][k].sigma_x = Pfront[i]
                        self.systolic_blocks1[self.x_nodes - d + i][j][k].sigma_y = Pfront[i]
                        self.systolic_blocks1[self.x_nodes - d + i][j][k].sigma_z = Pfront[i]

        if y_side == 'L':
            for j in range(d):
                for i in range(self.x_nodes):
                    for k in range(self.z_nodes):
                        self.systolic_blocks1[i][j][k].sigma_x = Pright[d - 1 - j]
                        self.systolic_blocks1[i][j][k].sigma_y = Pright[d - 1 - j]
                        self.systolic_blocks1[i][j][k].sigma_z = Pright[d - 1 - j]
        else:
            for j in range(d):
                for i in range(self.x_nodes):
                    for k in range(self.z_nodes):
                        self.systolic_blocks1[i][self.y_nodes - d + j][k].sigma_x = Pright[j]
                        self.systolic_blocks1[i][self.y_nodes - d + j][k].sigma_y = Pright[j]
                        self.systolic_blocks1[i][self.y_nodes - d + j][k].sigma_z = Pright[j]

        if z_side == 'B':
            for k in range(d):
                for i in range(self.x_nodes):
                    for j in range(self.y_nodes):
                        self.systolic_blocks1[i][j][k].sigma_x = Ptop[d - 1 - k]
                        self.systolic_blocks1[i][j][k].sigma_y = Ptop[d - 1 - k]
                        self.systolic_blocks1[i][j][k].sigma_z = Ptop[d - 1 - k]
        else:
            for k in range(d):
                for i in range(self.x_nodes):
                    for j in range(self.y_nodes):
                        self.systolic_blocks1[i][j][self.z_nodes - d + k].sigma_x = Ptop[k]
                        self.systolic_blocks1[i][j][self.z_nodes - d + k].sigma_y = Ptop[k]
                        self.systolic_blocks1[i][j][self.z_nodes - d + k].sigma_z = Ptop[k]

        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_parameters()

    def export_value_Ez(self):
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.Ez[i][j][k] = self.systolic_blocks1[i][j][k].Ez
        return self.Ez
