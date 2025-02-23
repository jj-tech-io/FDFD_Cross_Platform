from matplotlib.patches import Circle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import config


class CalcPML3D:
    """
    CalcPML3D - Computes 3D PML scaling factors for a Yee grid.
    
    This class supports three geometries:
      - 'cartesian'  : Standard Cartesian PML (applied independently in x, y, and z).
      - 'cylindrical': In the x–y plane the scaling factors depend on the radial coordinate
                       (with the physical region defined as an inscribed circle), while the z–direction
                       is treated as a robust Cartesian PML.
      - 'spherical'  : All three directions are scaled using the full 3D radial coordinate.
    
    Inputs:
       NGRID    : (Nx, Ny, Nz) grid dimensions.
       NPML     : (NXLO, NXHI, NYLO, NYHI, NZLO, NZHI) thickness in cells.
       RES      : (dx, dy, dz) cell sizes.
       BC       : (xbc, ybc, zbc) boundary conditions (0 => Dirichlet; 1 => Periodic).
       kinc     : (optional) incident wavevector (ignored here).
       geometry : string, one of 'cartesian', 'cylindrical', or 'spherical' (default: 'cartesian').
    """
    def __init__(self, NGRID, NPML, RES, BC=(0,0,0), kinc=None, normalize=False, geometry='cartesian',amax=3.0,cmax=1.0,p=2.0):
        self.geometry = geometry.lower()
        self.Nx, self.Ny, self.Nz = NGRID
        self.NXLO, self.NXHI, self.NYLO, self.NYHI, self.NZLO, self.NZHI = NPML
        self.dx, self.dy, self.dz = RES
        self.BC = BC  # apply PML only where BC[d]==0 (Dirichlet)
        self.kinc = kinc  # now ignored
        # PML parameters – adjust these as needed:
        self.amax = amax   # Maximum amplitude factor
        self.cmax = cmax   # Increase this to boost absorption (tune as needed)
        self.p    = p      # Ramp power (or increase to steepen the profile)
        self.eta0 = 377.0  # Free-space impedance

        shape = (self.Nx, self.Ny, self.Nz)
        self.sx = np.ones(shape, dtype=np.complex128)
        self.sy = np.ones(shape, dtype=np.complex128)
        self.sz = np.ones(shape, dtype=np.complex128)
        self._build_pml()

    def _build_pml(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        amax, cmax, p, eta0 = self.amax, self.cmax, self.p, self.eta0

        # Here we remove any dependence on kinc. In other words, the "angle correction"
        # factors are set to 1 in every direction.
        angle_corr_x = 1.0
        angle_corr_y = 1.0
        angle_corr_z = 1.0

        if self.geometry == 'cartesian':
            # X-direction:
            if self.BC[0] == 0:
                if self.NXLO > 0:
                    for nx in range(1, self.NXLO + 1):
                        ax = 1 + (amax - 1) * ((nx / self.NXLO) ** p)
                        cx = cmax * (np.sin(0.5 * np.pi * (nx / self.NXLO))) ** 2
                        i = self.NXLO - nx
                        self.sx[i, :, :] = ax * (1 - 1j * eta0 * cx * angle_corr_x)
                if self.NXHI > 0:
                    for nx in range(1, self.NXHI + 1):
                        ax = 1 + (amax - 1) * ((nx / self.NXHI) ** p)
                        cx = cmax * (np.sin(0.5 * np.pi * (nx / self.NXHI))) ** 2
                        i = (Nx - self.NXHI) + (nx - 1)
                        self.sx[i, :, :] = ax * (1 - 1j * eta0 * cx * angle_corr_x)
            # Y-direction:
            if self.BC[1] == 0:
                if self.NYLO > 0:
                    for ny in range(1, self.NYLO + 1):
                        ay = 1 + (amax - 1) * ((ny / self.NYLO) ** p)
                        cy = cmax * (np.sin(0.5 * np.pi * (ny / self.NYLO))) ** 2
                        j = self.NYLO - ny
                        self.sy[:, j, :] = ay * (1 - 1j * eta0 * cy * angle_corr_y)
                if self.NYHI > 0:
                    for ny in range(1, self.NYHI + 1):
                        ay = 1 + (amax - 1) * ((ny / self.NYHI) ** p)
                        cy = cmax * (np.sin(0.5 * np.pi * (ny / self.NYHI))) ** 2
                        j = (Ny - self.NYHI) + (ny - 1)
                        self.sy[:, j, :] = ay * (1 - 1j * eta0 * cy * angle_corr_y)
            # Z-direction:
            if self.BC[2] == 0:
                # Lower z boundary:
                if self.NZLO > 0:
                    for k in range(self.NZLO):
                        s_val = (self.NZLO - k) / self.NZLO
                        az = 1 + (amax - 1) * (s_val ** p)
                        cz = cmax * (np.sin(0.5 * np.pi * s_val)) ** 2
                        self.sz[:, :, k] = az * (1 - 1j * eta0 * cz * angle_corr_z)
                # Upper z boundary:
                if self.NZHI > 0:
                    for k in range(self.NZHI):
                        s_val = (k + 1) / self.NZHI
                        az = 1 + (amax - 1) * (s_val ** p)
                        cz = cmax * (np.sin(0.5 * np.pi * s_val)) ** 2
                        self.sz[:, :, (Nz - self.NZHI) + k] = az * (1 - 1j * eta0 * cz * angle_corr_z)

        # (The cylindrical and spherical branches would be modified similarly.)
        elif self.geometry == 'cylindrical':
            # Here we remove kinc-based corrections and use the nominal profile.
            x = np.linspace(0, (Nx - 1) * self.dx, Nx)
            y = np.linspace(0, (Ny - 1) * self.dy, Ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            center_x = (x[0] + x[-1]) / 2.0
            center_y = (y[0] + y[-1]) / 2.0
            R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            R0 = min(x[-1] - x[0], y[-1] - y[0]) / 2.0
            L_pml = self.NXHI * self.dx  # transverse PML thickness
            for i in range(Nx):
                for j in range(Ny):
                    r = R[i, j]
                    if r >= R0:
                        xi = (r - R0) / L_pml
                        xi = min(xi, 1.0)
                        a_r = 1 + (amax - 1) * (xi ** p)
                        c_r = cmax * (np.sin(0.5 * np.pi * xi)) ** 2
                        s_val = a_r * (1 - 1j * eta0 * c_r)  # no additional correction
                    else:
                        s_val = 1.0
                    self.sx[i, j, :] = s_val
                    self.sy[i, j, :] = s_val
            if self.BC[2] == 0:
                if self.NZLO > 0:
                    for k in range(self.NZLO):
                        s_val = (self.NZLO - k) / self.NZLO
                        az = 1 + (amax - 1) * (s_val ** p)
                        cz = cmax * (np.sin(0.5 * np.pi * s_val)) ** 2
                        self.sz[:, :, k] = az * (1 - 1j * eta0 * cz)
                if self.NZHI > 0:
                    for k in range(self.NZHI):
                        s_val = (k + 1) / self.NZHI
                        az = 1 + (amax - 1) * (s_val ** p)
                        cz = cmax * (np.sin(0.5 * np.pi * s_val)) ** 2
                        self.sz[:, :, (Nz - self.NZHI) + k] = az * (1 - 1j * eta0 * cz)
        elif self.geometry == 'spherical':
            x = np.linspace(0, (Nx - 1) * self.dx, Nx)
            y = np.linspace(0, (Ny - 1) * self.dy, Ny)
            z = np.linspace(0, (Nz - 1) * self.dz, Nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            center_x = (x[0] + x[-1]) / 2.0
            center_y = (y[0] + y[-1]) / 2.0
            center_z = (z[0] + z[-1]) / 2.0
            R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2 + (Z - center_z) ** 2)
            L_pml = self.NZHI * self.dz
            r_max = np.max(R)
            r_pml_start = r_max - L_pml
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        r = R[i, j, k]
                        if r >= r_pml_start:
                            xi = (r - r_pml_start) / L_pml
                            xi = min(xi, 1.0)
                            a_r = 1 + (amax - 1) * (xi ** p)
                            c_r = cmax * (np.sin(0.5 * np.pi * xi)) ** 2
                            s_val = a_r * (1 - 1j * eta0 * c_r)
                        else:
                            s_val = 1.0
                        self.sx[i, j, k] = s_val
                        self.sy[i, j, k] = s_val
                        self.sz[i, j, k] = s_val
        else:
            raise ValueError("Unknown geometry type. Choose 'cartesian', 'cylindrical', or 'spherical'.")

    def get_pml(self):
        return self.sx, self.sy, self.sz

    def visualize_pml(self):
        # (Visualization code unchanged)
        x = np.linspace(0, (self.Nx - 1) * self.dx, self.Nx)
        y = np.linspace(0, (self.Ny - 1) * self.dy, self.Ny)
        z = np.linspace(0, (self.Nz - 1) * self.dz, self.Nz)
        if self.geometry.lower() == 'spherical' and self.NZHI < self.Nz:
            z_index = self.Nz - self.NZHI
        else:
            z_index = self.Nz // 2
        mid_x = self.Nx // 2
        mid_y = self.Ny // 2
        vmin = np.min([np.min(np.abs(self.sx)),
                       np.min(np.abs(self.sy)),
                       np.min(np.abs(self.sz))])
        vmax = np.max([np.max(np.abs(self.sx)),
                       np.max(np.abs(self.sy)),
                       np.max(np.abs(self.sz))])

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        plane_xy = np.abs(self.sx[:, :, z_index])
        im0 = axs[0].imshow(plane_xy.T, origin='lower',
                            extent=[x[0], x[-1], y[0], y[-1]],
                            cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_xlabel("x (m)")
        axs[0].set_ylabel("y (m)")
        axs[0].set_title("z = {:.2e} m".format(z[z_index]))
        if self.geometry.lower() == 'cartesian':
            if self.BC[0] == 0:
                axs[0].axvline(x[self.NXLO], color='k', linestyle='--')
                axs[0].axvline(x[-(self.NXHI + 1)], color='k', linestyle='--')
            if self.BC[1] == 0:
                axs[0].axhline(y[self.NYLO], color='k', linestyle='--')
                axs[0].axhline(y[-(self.NYHI + 1)], color='k', linestyle='--')
        elif self.geometry.lower() == 'cylindrical':
            R0 = min(x[-1] - x[0], y[-1] - y[0]) / 2.0
            center = ((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0)
            circle = Circle(center, R0, edgecolor='k', facecolor='none', linestyle='--', linewidth=2)
            axs[0].add_patch(circle)
        elif self.geometry.lower() == 'spherical':
            center = ((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0)
            X_full, Y_full, Z_full = np.meshgrid(x, y, z, indexing='ij')
            R_full = np.sqrt((X_full - center[0]) ** 2 + (Y_full - center[1]) ** 2 +
                             (Z_full - (z[0] + z[-1]) / 2.0) ** 2)
            L_pml = self.NZHI * self.dz
            r_pml_start = np.max(R_full) - L_pml
            dz_eff = z[z_index] - (z[0] + z[-1]) / 2.0
            r_circle = np.sqrt(max(r_pml_start ** 2 - dz_eff ** 2, 0))
            
            circle = Circle(center, r_circle, edgecolor='k', facecolor='none', linestyle='--', linewidth=2)
            axs[0].add_patch(circle)
        fig.colorbar(im0, ax=axs[0])

        plane_xz = np.abs(self.sz[:, mid_y, :])
        im1 = axs[1].imshow(plane_xz.T, origin='lower',
                            extent=[x[0], x[-1], z[0], z[-1]],
                            cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].set_xlabel("x (m)")
        axs[1].set_ylabel("z (m)")
        axs[1].set_title("y = {:.2e} m".format(y[mid_y]))
        if self.BC[0] == 0:
            axs[1].axvline(x[self.NXLO], color='k', linestyle='--')
            axs[1].axvline(x[-(self.NXHI + 1)], color='k', linestyle='--')
        if self.BC[2] == 0:
            axs[1].axhline(z[self.NZLO], color='k', linestyle='--')
            axs[1].axhline(z[-(self.NZHI + 1)], color='k', linestyle='--')
        fig.colorbar(im1, ax=axs[1])

        plane_yz = np.abs(self.sz[mid_x, :, :])
        im2 = axs[2].imshow(plane_yz.T, origin='lower',
                            extent=[y[0], y[-1], z[0], z[-1]],
                            cmap='viridis', vmin=vmin, vmax=vmax)
        axs[2].set_xlabel("y (m)")
        axs[2].set_ylabel("z (m)")
        axs[2].set_title("x = {:.2e} m".format(x[mid_x]))
        if self.BC[1] == 0:
            axs[2].axvline(y[self.NYLO], color='k', linestyle='--')
            axs[2].axvline(y[-(self.NYHI + 1)], color='k', linestyle='--')
        if self.BC[2] == 0:
            axs[2].axhline(z[self.NZLO], color='k', linestyle='--')
            axs[2].axhline(z[-(self.NZHI + 1)], color='k', linestyle='--')
        fig.colorbar(im2, ax=axs[2])

        fig.suptitle("PML Absorption Profiles (" + self.geometry.capitalize() + ")", fontsize=18)
        plt.tight_layout()
        save_path = config.SAVE_DIR + "pml_absorption.png"
        plt.savefig(save_path, dpi=300)