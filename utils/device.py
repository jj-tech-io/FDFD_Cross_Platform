"""
Modular FDFD Solver for 3D Dielectric Structures
Author: Joel Johnson (2021)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import cupy as cp
import sys
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpx_linalg
from tqdm import tqdm

# Make sure that the utils folder is in your PYTHONPATH:
sys.path.append(r"C:\Users\joeli\Dropbox\Code\Mat_Solve\utils")

def build_cyl_epsilon(Nx, Ny, Nz, dx, dy, dz,
                      radius,
                      n_bg,
                      n_cyl_real, n_cyl_imag,
                      sub_n=4):
    """
    3D permittivity array for a z‐oriented cylinder, using subpixel sampling.
    The cylinder (of radius 'radius') is centered at (0,0,0).
    """
    eps_bg_sq  = n_bg**2
    n_cyl      = n_cyl_real + 1j*n_cyl_imag
    eps_cyl_sq = n_cyl**2

    eps_3d = eps_bg_sq * np.ones((Nx, Ny, Nz), dtype=np.complex128)

    # Build cell-centered coordinate arrays.
    xvals = np.atleast_1d((np.arange(Nx) - (Nx/2 - 0.5)) * dx)
    yvals = np.atleast_1d((np.arange(Ny) - (Ny/2 - 0.5)) * dy)
    zvals = np.atleast_1d((np.arange(Nz) - (Nz/2 - 0.5)) * dz)
    r_sq  = radius**2

    offsets = (np.arange(sub_n) + 0.5) / sub_n - 0.5
    for i, cx in enumerate(xvals):
        for j, cy in enumerate(yvals):
            # Only check cells that are near the cylinder.
            r_xy_sq_cell = cx**2 + cy**2
            margin = (dx/2)**2 + (dy/2)**2
            if r_xy_sq_cell > (radius + np.sqrt(margin))**2:
                continue
            for k, cz in enumerate(zvals):
                inside_count = 0
                for ox in offsets:
                    x_sub = cx + ox * dx
                    for oy in offsets:
                        y_sub = cy + oy * dy
                        for oz in offsets:
                            # For a cylinder only the x and y coordinates matter.
                            if x_sub**2 + y_sub**2 <= r_sq:
                                inside_count += 1
                frac = inside_count / (sub_n**3)
                eps_3d[i, j, k] = frac * eps_cyl_sq + (1 - frac) * eps_bg_sq

    return eps_3d

# def build_cyl_epsilon_tapered(Nx, Ny, Nz, dx, dy, dz,
#                               radius,
#                               n_bg,
#                               n_cyl_real, n_cyl_imag,
#                               NPML,  # tuple: (NXLO, NXHI, NYLO, NYHI, NZLO, NZHI)
#                               sub_n=4,
#                               amax=1.0,
#                               cmax=1.0,
#                               p=2.0,
#                               eta0=377.0):
#     """
#     Build a 3D relative permittivity array for a z‐oriented cylinder using subpixel sampling,
#     then taper the permittivity in the NPML regions (x, y, and z) so that it transitions smoothly
#     to the background value (assumed to be n_bg^2, typically 1.0). This ensures a constant impedance
#     in the NPML.

#     The effective permittivity is set as:
#         eps_eff = (1 - f) * eps_device + f * eps_bg,
#     where f increases from 0 (at the NPML interface) to 1 (at the outer boundary).

#     Inputs:
#       Nx, Ny, Nz   : Grid dimensions.
#       dx, dy, dz   : Grid spacings.
#       radius       : Cylinder radius.
#       n_bg         : Background refractive index (typically 1.0).
#       n_cyl_real, n_cyl_imag: Cylinder refractive index components.
#       NPML         : Tuple (NXLO, NXHI, NYLO, NYHI, NZLO, NZHI) NPML thickness (cells) in each direction.
#       sub_n        : Number of subpixel samples.

#     Returns:
#       eps_3d : 3D numpy array of tapered relative permittivity (shape (Nx, Ny, Nz)).
#     """
#     # Background and device permittivities.
#     eps_bg_sq  = n_bg**2
#     n_cyl = n_cyl_real + 1j * n_cyl_imag
#     eps_cyl_sq = n_cyl**2

#     # Build the "raw" permittivity using subpixel sampling.
#     eps_3d = eps_bg_sq * np.ones((Nx, Ny, Nz), dtype=np.complex128)
#     xvals = (np.arange(Nx) - (Nx/2 - 0.5)) * dx
#     yvals = (np.arange(Ny) - (Ny/2 - 0.5)) * dy
#     zvals = (np.arange(Nz) - (Nz/2 - 0.5)) * dz
#     r_sq = radius**2

#     offsets = (np.arange(sub_n) + 0.5) / sub_n - 0.5
#     for i, cx in enumerate(xvals):
#         for j, cy in enumerate(yvals):
#             # Skip cells that are clearly outside the cylinder.
#             if cx**2 + cy**2 > (radius + np.sqrt((dx/2)**2 + (dy/2)**2))**2:
#                 continue
#             for k, cz in enumerate(zvals):
#                 inside_count = 0
#                 for ox in offsets:
#                     x_sub = cx + ox * dx
#                     for oy in offsets:
#                         y_sub = cy + oy * dy
#                         for oz in offsets:
#                             if x_sub**2 + y_sub**2 <= r_sq:
#                                 inside_count += 1
#                 frac = inside_count / (sub_n**3)
#                 eps_3d[i, j, k] = frac * eps_cyl_sq + (1 - frac) * eps_bg_sq

#     # Taper the permittivity in the NPML regions.
#     NXLO, NXHI, NYLO, NYHI, NZLO, NZHI = NPML

#     # --- Taper in x-direction ---
#     for i in range(NXLO):
#         # f increases from 0 at the interface (i = NXLO-1) to 1 at the boundary (i = 0)
#         f = (i + 1) / NXLO
#         eps_3d[i, :, :] = (1 - f) * eps_3d[i, :, :] + f * eps_bg_sq
#     for i in range(NXHI):
#         f = (i + 1) / NXHI
#         eps_3d[Nx - NXHI + i, :, :] = (1 - f) * eps_3d[Nx - NXHI + i, :, :] + f * eps_bg_sq

#     # --- Taper in y-direction ---
#     for j in range(NYLO):
#         f = (j + 1) / NYLO
#         eps_3d[:, j, :] = (1 - f) * eps_3d[:, j, :] + f * eps_bg_sq
#     for j in range(NYHI):
#         f = (j + 1) / NYHI
#         eps_3d[:, Ny - NYHI + j, :] = (1 - f) * eps_3d[:, Ny - NYHI + j, :] + f * eps_bg_sq

#     # --- Taper in z-direction ---
#     # Lower z NPML: indices 0 to NZLO-1, with index 0 (outermost) -> f=1, index NZLO-1 (interface) -> f=0.
#     for k in range(NZLO):
#         if NZLO > 1:
#             f = 1 - (k / (NZLO - 1))
#         else:
#             f = 1
#         eps_3d[:, :, k] = (1 - f) * eps_3d[:, :, k] + f * eps_bg_sq
#     # Upper z NPML: indices Nz - NZHI to Nz-1, with index Nz - NZHI (interface) -> f=0, index Nz-1 (outermost) -> f=1.
#     for k in range(NZHI):
#         # For the upper z NPML.
#         f = (k + 1) / NZHI
#         eps_3d[:, :, Nz - NZHI + k] = (1 - f) * eps_3d[:, :, Nz - NZHI + k] + f * eps_bg_sq

#     return eps_3d
import numpy as np

def build_cyl_epsilon_tapered(Nx, Ny, Nz, dx, dy, dz,
                              radius,
                              n_bg,
                              n_cyl_real, n_cyl_imag,
                              NPML,  # tuple: (NXLO, NXHI, NYLO, NYHI, NZLO, NZHI)
                              sub_n=4,
                              amax=3.0,   # set to 3.0 (not 1.0) so that the taper is nontrivial
                              cmax=1.0,
                              p=2.0,
                              eta0=377.0):
    """
    Build a 3D relative permittivity array for a z‐oriented cylinder using subpixel sampling,
    then taper the permittivity in the NPML regions so that it transitions smoothly
    from the device value to the background value (n_bg^2). This helps maintain a constant 
    effective impedance through the NPML.
    
    The taper is computed using a normalized coordinate ξ (in each NPML region) and
    the following formulas:
    
         a(ξ) = 1 + (amax - 1) * ξ^p,
         c(ξ) = cmax * sin²(0.5πξ),
         s(ξ) = a(ξ) * [1 - i * eta0 * c(ξ)],
         f(ξ) = |s(ξ) - 1| / |s(1) - 1|.
    
    The effective permittivity is then:
    
         ε_eff = (1 - f(ξ)) * ε_device + f(ξ) * ε_bg.
    
    Inputs:
      Nx, Ny, Nz        : Grid dimensions.
      dx, dy, dz        : Grid spacings.
      radius            : Cylinder radius.
      n_bg              : Background refractive index.
      n_cyl_real,n_cyl_imag : Cylinder refractive index components.
      NPML              : Tuple (NXLO, NXHI, NYLO, NYHI, NZLO, NZHI) specifying NPML thickness (cells).
      sub_n             : Number of subpixel samples.
      amax, cmax, p, eta0: Taper parameters.
      
    Returns:
      eps_3d_tapered    : 3D numpy array of tapered relative permittivity (shape (Nx,Ny,Nz)).
    """
    # --- Compute background and device permittivities ---
    eps_bg_sq = n_bg**2
    n_cyl = n_cyl_real + 1j * n_cyl_imag
    eps_cyl_sq = n_cyl**2

    # --- Build raw device permittivity via subpixel sampling ---
    eps_3d = eps_bg_sq * np.ones((Nx, Ny, Nz), dtype=np.complex128)
    xvals = (np.arange(Nx) - (Nx/2 - 0.5)) * dx
    yvals = (np.arange(Ny) - (Ny/2 - 0.5)) * dy
    zvals = (np.arange(Nz) - (Nz/2 - 0.5)) * dz
    r_sq = radius**2

    offsets = (np.arange(sub_n) + 0.5) / sub_n - 0.5
    for i, cx in enumerate(xvals):
        for j, cy in enumerate(yvals):
            # Only process cells that might be inside the cylinder.
            if cx**2 + cy**2 > (radius + np.sqrt((dx/2)**2 + (dy/2)**2))**2:
                continue
            for k, cz in enumerate(zvals):
                inside_count = 0
                for ox in offsets:
                    x_sub = cx + ox * dx
                    for oy in offsets:
                        y_sub = cy + oy * dy
                        for oz in offsets:
                            if x_sub**2 + y_sub**2 <= r_sq:
                                inside_count += 1
                frac = inside_count / (sub_n**3)
                eps_3d[i, j, k] = frac * eps_cyl_sq + (1 - frac) * eps_bg_sq

    # --- Compute normalized coordinates (ξ) for each axis ---
    NXLO, NXHI, NYLO, NYHI, NZLO, NZHI = NPML

    # For x-direction:
    xi_x = np.zeros(Nx)
    if NXLO > 1:
        for i in range(NXLO):
            xi_x[i] = 1 - i / (NXLO - 1)
    if NXHI > 1:
        for i in range(Nx - NXHI, Nx):
            xi_x[i] = (i - (Nx - NXHI)) / (NXHI - 1)
    # For y-direction:
    xi_y = np.zeros(Ny)
    if NYLO > 1:
        for j in range(NYLO):
            xi_y[j] = 1 - j / (NYLO - 1)
    if NYHI > 1:
        for j in range(Ny - NYHI, Ny):
            xi_y[j] = (j - (Ny - NYHI)) / (NYHI - 1)
    # For z-direction:
    xi_z = np.zeros(Nz)
    if NZLO > 1:
        for k in range(NZLO):
            xi_z[k] = 1 - k / (NZLO - 1)
    if NZHI > 1:
        for k in range(Nz - NZHI, Nz):
            xi_z[k] = (k - (Nz - NZHI)) / (NZHI - 1)

    # Broadcast these 1D arrays into 3D.
    Xi_x, Xi_y, Xi_z = np.meshgrid(xi_x, xi_y, xi_z, indexing='ij')
    # Combine them via maximum (so that if any axis is in NPML, the cell is tapered).
    Xi = np.maximum(np.maximum(Xi_x, Xi_y), Xi_z)

    # --- Define the taper factor function ---
    def taper_factor(xi):
        # Compute the amplitude and conductivity scaling factors.
        a_val = 1 + (amax - 1) * (xi ** p)
        c_val = cmax * (np.sin(0.5 * np.pi * xi))**2
        s_val = a_val * (1 - 1j * eta0 * c_val)
        # Compute s(1) (outer edge)
        s_max = 1 + (amax - 1) * (1 ** p) * (1 - 1j * eta0 * cmax)
        denom = np.abs(s_max - 1)
        # Guard against division by (near) zero.
        if denom < 1e-12:
            return np.zeros_like(xi)
        return np.abs(s_val - 1) / denom

    F_eff = taper_factor(Xi)

    # --- Blend the raw permittivity with the background ---
    eps_3d_tapered = (1 - F_eff) * eps_3d + F_eff * eps_bg_sq

    return eps_3d_tapered
