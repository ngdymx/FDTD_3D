import numpy as np


class fdtd_3d_systolic_block:
    def __init__(self):
        self.Hz = 0
        self.Hzx = 0
        self.Hzy = 0
        self.Hx = 0
        self.Hxy = 0
        self.Hxz = 0
        self.Hy = 0
        self.Hyx = 0
        self.Hyz = 0
        self.Ez = 0
        self.Ezx = 0
        self.Ezy = 0
        self.Ex = 0
        self.Exy = 0
        self.Exz = 0
        self.Ey = 0
        self.Eyx = 0
        self.Eyz = 0
        self.dt = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.ep = 0
        self.mu = 0
        self.sigma_x = 0
        self.sigma_y = 0
        self.sigma_z = 0
        self.kex1 = 0
        self.kex2 = 0
        self.kex3 = 0
        self.key1 = 0
        self.key2 = 0
        self.key3 = 0
        self.kez1 = 0
        self.kez2 = 0
        self.kez3 = 0
        self.khx1 = 0
        self.khx2 = 0
        self.khx3 = 0
        self.khy1 = 0
        self.khy2 = 0
        self.khy3 = 0
        self.khz1 = 0
        self.khz2 = 0
        self.khz3 = 0

    def set_block(self, dt, dx, dy, dz, ep, mu, sigma_x=0, sigma_y=0, sigma_z=0):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ep = ep
        self.mu = mu
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.kex1 = (ep - 0.5 * sigma_x * dt) / (ep + 0.5 * sigma_x * dt)
        self.kex2 = dt / dy / (ep + 0.5 * sigma_x * dt)
        self.kex3 = dt / dz / (ep + 0.5 * sigma_x * dt)
        self.key1 = (ep - 0.5 * sigma_y * dt) / (ep + 0.5 * sigma_y * dt)
        self.key2 = dt / dz / (ep + 0.5 * sigma_y * dt)
        self.key3 = dt / dx / (ep + 0.5 * sigma_y * dt)
        self.kez1 = (ep - 0.5 * sigma_y * dt) / (ep + 0.5 * sigma_y * dt)
        self.kez2 = dt / dx / (ep + 0.5 * sigma_y * dt)
        self.kez3 = dt / dy / (ep + 0.5 * sigma_y * dt)

        self.khx1 = (mu - 0.5 * sigma_x * mu / ep * dt) / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.khx2 = dt / dz / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.khx3 = dt / dy / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.khy1 = (mu - 0.5 * sigma_y * mu / ep * dt) / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.khy2 = dt / dx / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.khy3 = dt / dz / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.khz1 = (mu - 0.5 * sigma_y * mu / ep * dt) / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.khz2 = dt / dy / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.khz3 = dt / dx / (mu + 0.5 * sigma_y * mu / ep * dt)

    def update_parameters(self):
        self.kex1 = (self.ep - 0.5 * self.sigma_x * self.dt) / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.kex2 = self.dt / self.dy / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.kex3 = self.dt / self.dz / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.key1 = (self.ep - 0.5 * self.sigma_y * self.dt) / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.key2 = self.dt / self.dz / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.key3 = self.dt / self.dx / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.kez1 = (self.ep - 0.5 * self.sigma_y * self.dt) / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.kez2 = self.dt / self.dx / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.kez3 = self.dt / self.dy / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.khx1 = (self.mu - 0.5 * self.sigma_x * self.mu / self.ep * self.dt) / \
                    (self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.khx2 = self.dt / self.dz / (self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.khx3 = self.dt / self.dy / (self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.khy1 = (self.mu - 0.5 * self.sigma_y * self.mu / self.ep * self.dt) / \
                    (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.khy2 = self.dt / self.dx / (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.khy3 = self.dt / self.dz / (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.khz1 = (self.mu - 0.5 * self.sigma_y * self.mu / self.ep * self.dt) / \
                    (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.khz2 = self.dt / self.dy / (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.khz3 = self.dt / self.dx / (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)

    def apply_src(self, value, stype='E'):
        if stype == 'E':
            self.Ez = self.Ez + value
        else:
            self.Hz = self.Hz + value

    def update_Hxz(self, Ey):
        self.Hxz = self.khx1 * self.Hxz + self.khx2 * (Ey - self.Ey)

    def update_Hxy(self, Ez):
        self.Hxy = self.khx1 * self.Hxy - self.khx3 * (Ez - self.Ez)

    def update_Hx(self):
        self.Hx = self.Hxz + self.Hxy

    def update_Hyz(self, Ex):
        self.Hyz = self.khy1 * self.Hyz - self.khy3 * (Ex - self.Ex)

    def update_Hyx(self, Ez):
        self.Hyx = self.khy1 * self.Hyx + self.khy2 * (Ez - self.Ez)

    def update_Hy(self):
        self.Hy = self.Hyz + self.Hyx

    def update_Hzy(self, Ex):
        self.Hzy = self.khz1 * self.Hzy + self.khz2 * (Ex - self.Ex)

    def update_Hzx(self, Ey):
        self.Hzx = self.khz1 * self.Hzx - self.khz3 * (Ey - self.Ey)

    def update_Hz(self):
        self.Hz = self.Hzy + self.Hzx

    def update_Exy(self, Hz):
        self.Exy = self.kex1 * self.Exy + self.kex2 * (self.Hz - Hz)

    def update_Exz(self, Hy):
        self.Exz = self.kex1 * self.Exz - self.kex3 * (self.Hy - Hy)

    def update_Ex(self):
        self.Ex = self.Exy + self.Exz

    def update_Eyz(self, Hx):
        self.Eyz = self.key1 * self.Eyz + self.key2 * (self.Hx - Hx)

    def update_Eyx(self, Hz):
        self.Eyx = self.key1 * self.Eyx - self.key3 * (self.Hz - Hz)

    def update_Ey(self):
        self.Ey = self.Eyz + self.Eyx

    def update_Ezx(self, Hy):
        self.Ezx = self.kez1 * self.Ezx + self.kez2 * (self.Hy - Hy)

    def update_Ezy(self, Hx):
        self.Ezy = self.kez1 * self.Ezy - self.kez3 * (self.Hx - Hx)

    def update_Ez(self):
        self.Ez = self.Ezx + self.Ezy


class FDTD_3D_space:
    def __init__(self, x_nodes, y_nodes, z_nodes, dt, dx, dy, dz, mu, ep):
        global col
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
        self.Hx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hxz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hxy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hyx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Hyz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ex = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Exy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Exz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ey = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Eyx = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Eyz = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ez = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ezy = np.zeros([x_nodes, y_nodes, z_nodes])
        self.Ezx = np.zeros([x_nodes, y_nodes, z_nodes])

    def apply_src(self, pos, value, stype='E'):
        x = pos[0]
        y = pos[1]
        z = pos[2]
        self.systolic_blocks1[x][y][z].apply_src(value, stype)

    def update(self):
        # update 3D Hxz
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    if k == self.z_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Hxz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Hxz(self.systolic_blocks1[i][j][k + 1].Ey)

        # update 3D Hxy
        for i in range(self.x_nodes):
            for k in range(self.z_nodes):
                for j in range(self.y_nodes):
                    if j == self.y_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Hxy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Hxy(self.systolic_blocks1[i][j + 1][k].Ez)

        # update 3D Hx
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Hx()

        # update 3D Hyx
        for j in range(self.y_nodes):
            for k in range(self.z_nodes):
                for i in range(self.x_nodes):
                    if i == self.x_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Hyx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Hyx(self.systolic_blocks1[i + 1][j][k].Ez)

        # update 3D Hyz
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    if k == self.z_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Hyz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Hyz(self.systolic_blocks1[i][j][k + 1].Ex)

        # update 3D Hy
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Hy()

        # update 3D Hzy
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    if j == self.z_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Hzy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Hzy(self.systolic_blocks1[i][j + 1][k].Ex)

        # update 3D Hzx
        for k in range(self.z_nodes):
            for j in range(self.y_nodes):
                for i in range(self.x_nodes):
                    if i == self.x_nodes - 1:
                        self.systolic_blocks1[i][j][k].update_Hzx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Hzx(self.systolic_blocks1[i + 1][j][k].Ey)

        # update 3D Hz
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Hz()

        # update 3D Exy
        for i in range(self.x_nodes):
            for k in range(self.z_nodes):
                for j in range(self.y_nodes):
                    if j == 0:
                        self.systolic_blocks1[i][j][k].update_Exy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Exy(self.systolic_blocks1[i][j - 1][k].Hz)

        # update 3D Exz
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    if k == 0:
                        self.systolic_blocks1[i][j][k].update_Exz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Exz(self.systolic_blocks1[i][j][k - 1].Hy)

        # update 3D Ex
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Ex()

        # update 3D Eyz
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                for k in range(self.z_nodes):
                    if k == 0:
                        self.systolic_blocks1[i][j][k].update_Eyz(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Eyz(self.systolic_blocks1[i][j][k - 1].Hx)

        # update 3D Eyx
        for j in range(self.y_nodes):
            for k in range(self.z_nodes):
                for i in range(self.x_nodes):
                    if i == 0:
                        self.systolic_blocks1[i][j][k].update_Eyx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Eyx(self.systolic_blocks1[i - 1][j][k].Hz)

        # update 3D Ey
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Ey()

        # update 3D Ezx
        for k in range(self.z_nodes):
            for j in range(self.y_nodes):
                for i in range(self.x_nodes):
                    if i == 0:
                        self.systolic_blocks1[i][j][k].update_Ezx(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Ezx(self.systolic_blocks1[i - 1][j][k].Hy)

        # update 3D Ezy
        for k in range(self.z_nodes):
            for i in range(self.x_nodes):
                for j in range(self.y_nodes):
                    if j == 0:
                        self.systolic_blocks1[i][j][k].update_Ezy(0)
                    else:
                        self.systolic_blocks1[i][j][k].update_Ezy(self.systolic_blocks1[i][j - 1][k].Hx)

        # update 3D Ez
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.systolic_blocks1[i][j][k].update_Ez()

    def export_value_Ez(self):
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                for k in range(self.z_nodes):
                    self.Ez[i][j][k] = self.systolic_blocks1[i][j][k].Ez
        return self.Ez
