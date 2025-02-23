import numpy as np

# def build_plane_wave(
#     k0,
#     Nx, Ny, Nz, dx, dy, dz,
#     lam0, theta_deg, phi_deg,
#     pol_angle_deg=0.0,
#     amplitude=1.0,
#     pml_cells=(0, 0, 0, 0, 0, 0),
#     padding=(0.0, 0.0, 0.0),
#     taper_width=(0.0, 0.0, 0.0),
#     phase_correct=False
# ):
#     """
#     Build an incident plane wave with optional tapering (e.g. to reduce edge effects near the PML).
#     """
#     # Convert angles to radians.
#     theta = np.deg2rad(theta_deg)
#     phi   = np.deg2rad(phi_deg)

#     n_bg = 1.0  
#     k_bg = k0 * n_bg

#     # Wavevector components.
#     kx = k_bg * np.sin(theta) * np.cos(phi)
#     ky = k_bg * np.sin(theta) * np.sin(phi)
#     kz = k_bg * np.cos(theta)

#     k_vec = np.array([kx, ky, kz], dtype=float)
#     k_hat = k_vec / np.linalg.norm(k_vec)

#     # Determine a polarization basis.
#     test_vec = np.array([0, 1, 0], dtype=float)
#     if abs(np.dot(test_vec, k_hat)) > 0.95:
#         test_vec = np.array([1, 0, 0], dtype=float)
#     test_vec = test_vec - np.dot(test_vec, k_hat) * k_hat
#     test_vec = test_vec / np.linalg.norm(test_vec)

#     s_perp = np.cross(k_hat, test_vec)
#     s_perp = s_perp / np.linalg.norm(s_perp)

#     alpha_pol = np.deg2rad(pol_angle_deg)
#     E_hat = np.cos(alpha_pol) * test_vec + np.sin(alpha_pol) * s_perp

#     # Create cell-centered coordinate arrays.
#     xvals = (np.arange(Nx) - Nx/2 + 0.5) * dx
#     yvals = (np.arange(Ny) - Ny/2 + 0.5) * dy
#     zvals = (np.arange(Nz) - Nz/2 + 0.5) * dz
#     X, Y, Z = np.meshgrid(xvals, yvals, zvals, indexing='ij')

#     # Optionally apply phase correction.
#     if phase_correct:
#         z_min_phys = - (Nz/2) * dz
#         L_phys = Nz * dz
#         Z_wrapped = ((Z - z_min_phys) % L_phys) + z_min_phys
#     else:
#         Z_wrapped = Z

#     phase = np.exp(1j * (kx * X + ky * Y + kz * Z_wrapped))

#     if phase_correct:
#         z_min = np.min(zvals)
#         correction = np.exp(-1j * kz * (Z - z_min))
#         phase = phase * correction

#     # Tapering near the PML.
#     half_x = 0.5 * Nx * dx
#     half_y = 0.5 * Ny * dy
#     half_z = 0.5 * Nz * dz

#     pml_x_lo, pml_x_hi, pml_y_lo, pml_y_hi, pml_z_lo, pml_z_hi = pml_cells
#     px_lo = pml_x_lo * dx
#     px_hi = pml_x_hi * dx
#     py_lo = pml_y_lo * dy
#     py_hi = pml_y_hi * dy
#     pz_lo = pml_z_lo * dz
#     pz_hi = pml_z_hi * dz

#     pml_thick_x = np.where(X >= 0, px_hi, px_lo)
#     pml_thick_y = np.where(Y >= 0, py_hi, py_lo)
#     pml_thick_z = np.where(Z >= 0, pz_hi, pz_lo)

#     dist_to_pml_x = half_x - np.abs(X) - pml_thick_x
#     dist_to_pml_y = half_y - np.abs(Y) - pml_thick_y
#     dist_to_pml_z = half_z - np.abs(Z) - pml_thick_z

#     pad_x, pad_y, pad_z = padding
#     tw_x, tw_y, tw_z = taper_width

#     dist_pad_x = dist_to_pml_x - pad_x
#     dist_pad_y = dist_to_pml_y - pad_y
#     dist_pad_z = dist_to_pml_z - pad_z

#     taper_mask_x = np.ones_like(dist_pad_x, dtype=float)
#     taper_mask_y = np.ones_like(dist_pad_y, dtype=float)
#     taper_mask_z = np.ones_like(dist_pad_z, dtype=float)

#     idx_x = dist_pad_x < 0
#     idx_y = dist_pad_y < 0
#     idx_z = dist_pad_z < 0

#     inside_x = -dist_pad_x[idx_x]
#     inside_y = -dist_pad_y[idx_y]
#     inside_z = -dist_pad_z[idx_z]

#     arg_x = np.minimum(inside_x / tw_x, 1.0, where=(tw_x > 0))
#     arg_y = np.minimum(inside_y / tw_y, 1.0, where=(tw_y > 0))
#     arg_z = np.minimum(inside_z / tw_z, 1.0, where=(tw_z > 0))

#     taper_mask_x[idx_x] = np.cos(arg_x * np.pi/2)**2
#     taper_mask_y[idx_y] = np.cos(arg_y * np.pi/2)**2
#     taper_mask_z[idx_z] = np.cos(arg_z * np.pi/2)**2

#     taper_mask = taper_mask_x * taper_mask_y * taper_mask_z

#     # Build the three components of the incident E–field.
#     Ex_3d = amplitude * E_hat[0] * phase * taper_mask
#     Ey_3d = amplitude * E_hat[1] * phase * taper_mask
#     Ez_3d = amplitude * E_hat[2] * phase * taper_mask

#     # Compute the total number of cells.
#     N_total = Nx * Ny * Nz

#     # Assemble into a flattened source vector.
#     fsrc = np.zeros(3 * N_total, dtype=np.complex128)
#     fsrc[0:N_total] = Ex_3d.ravel()
#     fsrc[N_total:2*N_total] = Ey_3d.ravel()
#     fsrc[2*N_total:3*N_total] = Ez_3d.ravel()

#     return fsrc



def build_plane_wave(
    k0,
    Nx, Ny, Nz, dx, dy, dz,
    lam0, theta_deg, phi_deg,
    pol_angle_deg=0.0,
    amplitude=1.0,
    phase_correct=False
):
    """
    Build a 3D incident plane wave field over the entire domain.
    This field is used as the incident source in the solver.
    """
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    n_bg = 1.0
    k_bg = k0 * n_bg
    kx = k_bg * np.sin(theta) * np.cos(phi)
    ky = k_bg * np.sin(theta) * np.sin(phi)
    kz = k_bg * np.cos(theta)
    
    # Polarization basis.
    k_vec = np.array([kx, ky, kz])
    k_hat = k_vec / np.linalg.norm(k_vec)
    test_vec = np.array([0, 1, 0], dtype=float)
    if abs(np.dot(test_vec, k_hat)) > 0.95:
        test_vec = np.array([1, 0, 0], dtype=float)
    test_vec = test_vec - np.dot(test_vec, k_hat)*k_hat
    test_vec = test_vec / np.linalg.norm(test_vec)
    s_perp = np.cross(k_hat, test_vec)
    s_perp = s_perp / np.linalg.norm(s_perp)
    alpha_pol = np.deg2rad(pol_angle_deg)
    E_hat = np.cos(alpha_pol)*test_vec + np.sin(alpha_pol)*s_perp

    x = (np.arange(Nx) - Nx/2 + 0.5)*dx
    y = (np.arange(Ny) - Ny/2 + 0.5)*dy
    z = (np.arange(Nz) - Nz/2 + 0.5)*dz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    if phase_correct:
        z_min_phys = - (Nz/2)*dz
        L_phys = Nz*dz
        Z_wrapped = ((Z - z_min_phys) % L_phys) + z_min_phys
    else:
        Z_wrapped = Z
    phase = np.exp(1j*(kx*X + ky*Y + kz*Z_wrapped))
    if phase_correct:
        z_min = np.min(z)
        correction = np.exp(-1j*kz*(Z-z_min))
        phase *= correction
    Ex = amplitude * E_hat[0] * phase
    Ey = amplitude * E_hat[1] * phase
    Ez = amplitude * E_hat[2] * phase

    # Compute total number of cells.
    N_total = Nx * Ny * Nz

    fsrc = np.zeros(3*N_total, dtype=np.complex128)
    fsrc[0:N_total] = Ex.ravel()
    fsrc[N_total:2*N_total] = Ey.ravel()
    fsrc[2*N_total:3*N_total] = Ez.ravel()
    return fsrc