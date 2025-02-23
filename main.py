"""
Modular FDFD Solver for 3D Dielectric Structures
Author: Joel Johnson (2025)
Modified to save an info.txt file and a giant CSV file with cell indices and column values.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpx_linalg
from tqdm import tqdm
import pandas as pd  # For saving results as CSV

# Make sure that the utils folder is in your PYTHONPATH:
sys.path.append(r"./utils")
import plot_utils
import yeeder, pml
import device
import source

import config
import importlib
importlib.reload(config)
importlib.reload(yeeder)
importlib.reload(pml)
importlib.reload(plot_utils)
importlib.reload(device)
importlib.reload(source)

from yeeder import yeeder3d
from pml import CalcPML3D
from plot_utils import plot_pml_slices, scatter_3d_field, plot_epsilon, plot_slices_with_pml
from device import build_cyl_epsilon_tapered
from source import build_plane_wave

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def curl_E_to_H(Ex, Ey, Ez, dx, dy, dz, omega, mu0):
    """
    Compute H = (1/(j ω μ0)) * (∇×E) from the electric field components.
    """
    dEx_dx, dEx_dy, dEx_dz = np.gradient(Ex, dx, dy, dz, edge_order=2)
    dEy_dx, dEy_dy, dEy_dz = np.gradient(Ey, dx, dy, dz, edge_order=2)
    dEz_dx, dEz_dy, dEz_dz = np.gradient(Ez, dx, dy, dz, edge_order=2)
    curlE_x = dEz_dy - dEy_dz
    curlE_y = dEx_dz - dEz_dx
    curlE_z = dEy_dx - dEx_dy
    Hx = (1/(1j * omega * mu0)) * curlE_x
    Hy = (1/(1j * omega * mu0)) * curlE_y
    Hz = (1/(1j * omega * mu0)) * curlE_z
    return Hx, Hy, Hz

def curl_E_to_H_tilde(Ex, Ey, Ez, dx, dy, dz, omega, mu0, eta0):
    """
    Compute H_tilde = -j*eta0*H, where H = (1/(j ω μ0))*(∇×E).
    """
    Hx, Hy, Hz = curl_E_to_H(Ex, Ey, Ez, dx, dy, dz, omega, mu0)
    return -1j * eta0 * Hx, -1j * eta0 * Hy, -1j * eta0 * Hz

def auto_vlims(*fields):
    """
    Compute common (vmin,vmax) limits based on the maximum absolute value among all provided fields.
    """
    maxval = 0
    for f in fields:
        maxval = max(maxval, np.max(np.abs(np.real(f))))
    return (-maxval, maxval)

def poynting_vector(Ex, Ey, Ez, Hx, Hy, Hz):
    """
    Compute the real Poynting vector S = 0.5*Re(E x H*).
    """
    Hx_conj = np.conj(Hx)
    Hy_conj = np.conj(Hy)
    Hz_conj = np.conj(Hz)
    Sx = 0.5 * np.real(Ey*Hz_conj - Ez*Hy_conj)
    Sy = 0.5 * np.real(Ez*Hx_conj - Ex*Hz_conj)
    Sz = 0.5 * np.real(Ex*Hy_conj - Ey*Hx_conj)
    return Sx, Sy, Sz

# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def main():
    #############################################################################
    # Simulation Constants and GPU/Results Setup
    #############################################################################
    # Clear previous GPU allocations and results folder
    cp.get_default_memory_pool().free_all_blocks()
    results_dir = config.SAVE_DIR
    for file in os.listdir(results_dir):
        os.remove(os.path.join(results_dir, file))
    
    c0       = 299792458.0                    # Speed of light (m/s)
    mu0      = 4e-7 * np.pi                   # Permeability (H/m)
    # Set vacuum permittivity:
    epsilon0 = 8.854187817e-12                # Permittivity (F/m)
    print(f"epsilon0 = {epsilon0:.3e} F/m")
    eta0     = np.sqrt(mu0/epsilon0)          # Impedance (~377 ohms)

    wavelength = 550e-9                     # Free-space wavelength (m)
    theta_deg  = 90.0                       # Polar angle (deg)
    phi_deg    = 45.0                        # Azimuth angle (deg)
    pol_angle_deg = 0.0                   # Polarization angle (deg)
    radius     = 1.0e-6                     # Cylinder radius (m)

    n_real     = 1.56                       # Cylinder refractive index (real)
    n_imag     = 0.007                      # Cylinder refractive index (imag)
    resolution = 15                         # Grid points per wavelength
    npml       = 15                         # PML size (cells)
    maxiters   = 1500                       # GMRES maximum iterations
    tol        = 1e-6                       # GMRES tolerance

    omega = 2*np.pi*c0 / wavelength         # Angular frequency
    k0 = omega / c0                         # Vacuum wavenumber
    print(f"omega = {omega:.3e} rad/s, initial k0 = {k0:.3e} m^-1")
    # Recompute k0 (should be equivalent)
    k0 = omega * np.sqrt(mu0*epsilon0)
    print(f"omega = {omega:.3e} rad/s, k0 = {k0:.3e} m^-1")
    dx = wavelength / resolution
    dy = dx
    dz = dx

    # ---------------------------
    # Domain Setup
    # ---------------------------
    domain = 3.0 * radius + 2.0 * npml * dx
    Nx = int(round(domain / dx))
    Ny = Nx
    Nz = Nx    
    theta = np.deg2rad(theta_deg)
    phi   = np.deg2rad(phi_deg)
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    # Zero out very small values:
    kx = 0 if abs(kx) < 1e-6 else kx
    ky = 0 if abs(ky) < 1e-6 else ky
    kz = 0 if abs(kz) < 1e-6 else kz
    print(f"kx = {kx:.3e} m^-1, ky = {ky:.3e} m^-1, kz = {kz:.3e} m^-1")
    print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}")
    NGRID = (Nx, Ny, Nz)
    RES = (dx, dy, dz)
    NPML  = (npml, npml, npml, npml, npml, npml)
    BC    = (0, 0, 0)  # (0: PML applied; 1: periodic)
    pml_geometry = 'cartesian'  # 'cylindrical' or 'rectangular'
    
    # ---------------------------
    # Build Incident Plane Wave Source
    # ---------------------------
    Einc_vec = build_plane_wave(
        k0,
        Nx, Ny, Nz, dx, dy, dz,
        wavelength, theta_deg, phi_deg,
        pol_angle_deg=pol_angle_deg,
        amplitude=1.0,
        phase_correct=False
    )
    
    # ---------------------------
    # Build Cylinder Permittivity (tapered)
    # ---------------------------
    amax = 4.0 # Maximum value of the taper
    cmax = 1.0 #
    p = 3.0   # Power of the taper
    eps_3d = build_cyl_epsilon_tapered(Nx, Ny, Nz, dx, dy, dz,
                                       radius, 1.0, n_real, n_imag,
                                       NPML, sub_n=4, amax=amax, cmax=cmax, p=p)
    # Ensure the taper never goes below free space (ε = 1)
    eps_3d = np.maximum(np.real(eps_3d), 1.0) + 1j * np.imag(eps_3d)
    print("min(eps) =", np.min(np.real(eps_3d)),
          "max(eps) =", np.max(np.real(eps_3d)))
    
    N_total = Nx * Ny * Nz
    # Define cell-centered coordinates.
    xvals = (np.arange(Nx) - Nx/2 + 0.5) * dx
    yvals = (np.arange(Ny) - Ny/2 + 0.5) * dy
    zvals = (np.arange(Nz) - Nz/2 + 0.5) * dz
    # Plot epsilon slices.
    plot_epsilon(eps_3d, xvals, yvals, zvals)
    
    # ---------------------------
    # Build Sparse Operators with PML
    # ---------------------------
    eps_diag = eps_3d.ravel()
    eps_mat  = sp.diags(eps_diag, 0, shape=(N_total, N_total), format='csc')
    eps_big  = sp.block_diag([eps_mat, eps_mat, eps_mat], format='csc')

    eps_bg_diag = np.ones(N_total, dtype=np.complex128)
    eps_bg_mat  = sp.diags(eps_bg_diag, 0, shape=(N_total, N_total), format='csc')
    eps_bg_big  = sp.block_diag([eps_bg_mat, eps_bg_mat, eps_bg_mat], format='csc')

    # Determine PML thickness in each direction based on BC.
    Nxlo = npml if BC[0] == 0 else 0
    Nxhi = npml if BC[0] == 0 else 0
    Nylo = npml if BC[1] == 0 else 0
    Nyhi = npml if BC[1] == 0 else 0
    Nzlo = npml if BC[2] == 0 else 0
    Nzhi = npml if BC[2] == 0 else 0

    pml = CalcPML3D(NGRID, NPML, RES, BC, kinc=np.array([kx, ky, kz]),
                    geometry=pml_geometry, amax=amax, cmax=cmax, p=p)
    sx, sy, sz = pml.get_pml()
    pml.visualize_pml()

    DEX, DEY, DEZ, DHX, DHY, DHZ = yeeder3d(NGRID, RES, BC, kinc=np.array([kx, ky, kz]))

    sx_flat = sx.ravel()
    sy_flat = sy.ravel()
    sz_flat = sz.ravel()
    Sx     = sp.diags(sx_flat, 0, format='csc')
    Sy     = sp.diags(sy_flat, 0, format='csc')
    Sz     = sp.diags(sz_flat, 0, format='csc')
    Sx_inv = sp.diags(1/sx_flat, 0, format='csc')
    Sy_inv = sp.diags(1/sy_flat, 0, format='csc')
    Sz_inv = sp.diags(1/sz_flat, 0, format='csc')

    DEX_s = Sx @ DEX @ Sx_inv
    DEY_s = Sy @ DEY @ Sy_inv
    DEZ_s = Sz @ DEZ @ Sz_inv

    CE = sp.bmat([
        [None,      -DEZ_s,    DEY_s],
        [DEZ_s,     None,      -DEX_s],
        [-DEY_s,    DEX_s,     None]
    ], format='csc')

    DHX_s = Sy @ DHX @ Sy_inv
    DHY_s = Sz @ DHY @ Sz_inv
    DHZ_s = Sx @ DHZ @ Sx_inv

    CH = sp.bmat([
        [None,     -DHZ_s,    DHY_s],
        [DHZ_s,    None,      -DHX_s],
        [-DHY_s,   DHX_s,     None]
    ], format='csc')

    alpha_coeff = -1j * eta0
    beta  = -1/(1j * eta0)
    CH_s = alpha_coeff * CH
    CE_s = beta  * CE

    A_actual = CH_s @ CE_s - (k0**2)*eps_big
    print(f"Shape of A: {A_actual.shape}")
    A_bg     = CH_s @ CE_s - (k0**2)*eps_bg_big

    # ---------------------------
    # Process Incident Field (apply PML masking/phase correction)
    # ---------------------------
    Ex_inc = Einc_vec[0:N_total].reshape((Nx, Ny, Nz))
    Ey_inc = Einc_vec[N_total:2*N_total].reshape((Nx, Ny, Nz))
    Ez_inc = Einc_vec[2*N_total:3*N_total].reshape((Nx, Ny, Nz))
    Hx_inc, Hy_inc, Hz_inc = curl_E_to_H(Ex_inc, Ey_inc, Ez_inc, dx, dy, dz, omega, mu0)
    incident_fields = {
        "Ex_inc": Ex_inc,
        "Ey_inc": Ey_inc,
        "Ez_inc": Ez_inc,
        "Hx_inc": Hx_inc,
        "Hy_inc": Hy_inc,
        "Hz_inc": Hz_inc
    }
    for field_name, field in incident_fields.items():
        plot_slices_with_pml(
            field, xvals, yvals, zvals,
            NPML[0], NPML[1], NPML[2], NPML[3], NPML[4], NPML[5],
            cmap='seismic_r',
            title=f"Incident {field_name}",
            x_slice=0, y_slice=0, z_slice=0,
            pml_geometry=pml_geometry, BC=BC
        )
        
    # Apply phase correction if needed.
    z_full = np.tile(zvals, (Nx, Ny, 1))
    z_min = np.min(zvals)
    correction = np.exp(-1j * kz * (z_full - z_min))
    Ex_inc = Ex_inc * correction
    Ey_inc = Ey_inc * correction
    Ez_inc = Ez_inc * correction
    Einc_vec[0:N_total] = Ex_inc.ravel()
    Einc_vec[N_total:2*N_total] = Ey_inc.ravel()
    Einc_vec[2*N_total:3*N_total] = Ez_inc.ravel()

    # ---------------------------
    # Build RHS and Solve for Scattered Field
    # ---------------------------
    rhs_vec = - (A_actual - A_bg) @ Einc_vec
    A_gpu   = cpsp.csc_matrix(A_actual)
    rhs_gpu = cp.asarray(rhs_vec)
    x0_gpu  = cp.zeros_like(rhs_gpu)

    print(f"Solving scattered-field system via GMRES: maxiters={maxiters}, tol={tol} ...")
    pbar = tqdm(total=maxiters, desc="GMRES Iters")
    def gmres_callback(resid):
        pbar.update(1)
        pbar.set_postfix_str(f"resid={resid:.8e}")
    x_sc_gpu, exitcode = cpx_linalg.gmres(
        A_gpu, rhs_gpu,
        x0=x0_gpu,
        tol=tol,
        maxiter=maxiters,
        restart=1,
        callback=gmres_callback,
        callback_type='pr_norm'
    )
    pbar.close()
    if exitcode == 0:
        print("GMRES converged!")
    else:
        print(f"GMRES ended with exit code={exitcode}")
    E_sc_vec = x_sc_gpu.get()

    Ex_sc = E_sc_vec[0:N_total].reshape((Nx, Ny, Nz))
    Ey_sc = E_sc_vec[N_total:2*N_total].reshape((Nx, Ny, Nz))
    Ez_sc = E_sc_vec[2*N_total:3*N_total].reshape((Nx, Ny, Nz))

    Ex_tot = Ex_inc + Ex_sc
    Ey_tot = Ey_inc + Ey_sc
    Ez_tot = Ez_inc + Ez_sc

    # ---------------------------
    # Compute H fields from E fields (and H_tilde)
    # ---------------------------
    Hx_inc, Hy_inc, Hz_inc = curl_E_to_H(Ex_inc, Ey_inc, Ez_inc, dx, dy, dz, omega, mu0)
    Hx_sc, Hy_sc, Hz_sc = curl_E_to_H(Ex_sc, Ey_sc, Ez_sc, dx, dy, dz, omega, mu0)
    Hx_tot, Hy_tot, Hz_tot = curl_E_to_H(Ex_tot, Ey_tot, Ez_tot, dx, dy, dz, omega, mu0)
    Hx_inc_tilde, Hy_inc_tilde, Hz_inc_tilde = curl_E_to_H_tilde(Ex_inc, Ey_inc, Ez_inc, dx, dy, dz, omega, mu0, eta0)
    Hx_sc_tilde, Hy_sc_tilde, Hz_sc_tilde = curl_E_to_H_tilde(Ex_sc, Ey_sc, Ez_sc, dx, dy, dz, omega, mu0, eta0)
    Hx_tot_tilde, Hy_tot_tilde, Hz_tot_tilde = curl_E_to_H_tilde(Ex_tot, Ey_tot, Ez_tot, dx, dy, dz, omega, mu0, eta0)

    # ---------------------------
    # Visualization of Fields (Full Domain with PML overlays)
    # ---------------------------
    vmin_e, vmax_e = auto_vlims(Ex_inc, Ex_sc, Ex_tot)
    field_dict = {
        "Ex_sc": Ex_sc,
        "Ex_tot": Ex_tot,
        "Ey_sc": Ey_sc,
        "Ey_tot": Ey_tot,
        "Ez_sc": Ez_sc,
        "Ez_tot": Ez_tot,
        "Hx_sc_tilde": Hx_sc_tilde,
        "Hx_tot_tilde": Hx_tot_tilde,
        "Hy_sc_tilde": Hy_sc_tilde,
        "Hy_tot_tilde": Hy_tot_tilde,
        "Hz_sc_tilde": Hz_sc_tilde,
        "Hz_tot_tilde": Hz_tot_tilde
    }
    
    for field_name, field in field_dict.items():
        plt.close('all')
        plot_slices_with_pml(
            field, xvals, yvals, zvals,
            NPML[0], NPML[1], NPML[2], NPML[3], NPML[4], NPML[5],
            cmap='seismic_r',
            title=field_name,
            vmin=vmin_e, vmax=vmax_e,
            x_slice=0, y_slice=0, z_slice=0,
            pml_geometry=pml_geometry, BC=BC
        )
    
    # (Optional) 3D scatter plots
    PLOT_3D = True
    if PLOT_3D:
        X_, Y_, Z_ = np.meshgrid(xvals, yvals, zvals, indexing='ij')
        E_amp = np.sqrt(np.abs(Ex_tot)**2 + np.abs(Ey_tot)**2 + np.abs(Ez_tot)**2)
        scatter_3d_field(X_, Y_, Z_, E_amp, stride=1, title="abs(E_tot)", cmap="jet", alpha=0.4)
        H_amp = np.sqrt(np.abs(Hx_tot)**2 + np.abs(Hy_tot)**2 + np.abs(Hz_tot)**2)
        scatter_3d_field(X_, Y_, Z_, H_amp, stride=1, title="abs(H_tot)", cmap="magma", alpha=0.4)
        Hx_conj = np.conj(Hx_tot)
        Hy_conj = np.conj(Hy_tot)
        Hz_conj = np.conj(Hz_tot)
        Sx = 0.5 * np.real(Ey_tot*Hz_conj - Ez_tot*Hy_conj)
        Sy = 0.5 * np.real(Ez_tot*Hx_conj - Ex_tot*Hz_conj)
        Sz = 0.5 * np.real(Ex_tot*Hy_conj - Ey_tot*Hx_conj)
        S_amp = np.sqrt(Sx**2 + Sy**2 + Sz**2)
        scatter_3d_field(X_, Y_, Z_, S_amp, stride=1, title="abs(S)", cmap="viridis", alpha=0.4)
    
    # =============================================================================
    # SAVE GIANT CSV FILE WITH ALL RESULTS (one row per cell)
    # =============================================================================
    print("Saving giant CSV file with all results...")
    # Compute scattered electric field (Escat) vector info from real parts.
    Escat_x = np.real(Ex_sc)
    Escat_y = np.real(Ey_sc)
    Escat_z = np.real(Ez_sc)
    Escat_mag = np.sqrt(Escat_x**2 + Escat_y**2 + Escat_z**2)
    Escat_theta = np.arccos(np.where(Escat_mag > 0, Escat_z/Escat_mag, 0))
    Escat_phi   = np.arctan2(Escat_y, Escat_x)
    
    # Compute scattered magnetic field (Hscat) vector info.
    Hscat_x = np.real(Hx_sc)
    Hscat_y = np.real(Hy_sc)
    Hscat_z = np.real(Hz_sc)
    Hscat_mag = np.sqrt(Hscat_x**2 + Hscat_y**2 + Hscat_z**2)
    Hscat_theta = np.arccos(np.where(Hscat_mag > 0, Hscat_z/Hscat_mag, 0))
    Hscat_phi   = np.arctan2(Hscat_y, Hscat_x)
    
    # Compute scattered Poynting vector (Sscat) from scattered fields.
    Sx_sc, Sy_sc, Sz_sc = poynting_vector(Ex_sc, Ey_sc, Ez_sc, Hx_sc, Hy_sc, Hz_sc)
    Sscat_mag = np.sqrt(Sx_sc**2 + Sy_sc**2 + Sz_sc**2)
    Sscat_theta = np.arccos(np.where(Sscat_mag > 0, Sz_sc/Sscat_mag, 0))
    Sscat_phi   = np.arctan2(Sy_sc, Sx_sc)
    
    # Create index arrays for each cell.
    indices = np.indices((Nx, Ny, Nz))
    x_idx = indices[0].ravel()
    y_idx = indices[1].ravel()
    z_idx = indices[2].ravel()
    total_cells = N_total  # equals Nx * Ny * Nz
    
    # Build the DataFrame.
    df = pd.DataFrame({
        "x_index": x_idx,
        "y_index": y_idx,
        "z_index": z_idx,
        "theta_i": np.full(total_cells, theta_deg),
        "phi_i": np.full(total_cells, phi_deg),
        "radius": np.full(total_cells, radius),
        "shape": np.full(total_cells, "circular"),
        "n_r": np.full(total_cells, n_real),
        "n_i": np.full(total_cells, n_imag),
        "lambda": np.full(total_cells, wavelength),
        "Escat_mag": Escat_mag.ravel(),
        "Escat_theta": Escat_theta.ravel(),
        "Escat_phi": Escat_phi.ravel(),
        "Hscat_mag": Hscat_mag.ravel(),
        "Hscat_theta": Hscat_theta.ravel(),
        "Hscat_phi": Hscat_phi.ravel(),
        "Sscat_mag": Sscat_mag.ravel(),
        "Sscat_theta": Sscat_theta.ravel(),
        "Sscat_phi": Sscat_phi.ravel()
    })
    
    csv_filename = os.path.join(results_dir, "all_results.csv")
    df.to_csv(csv_filename, index=False)
    print("Giant CSV file saved to:", csv_filename)
    
    # =============================================================================
    # SAVE INFO FILE WITH SIMULATION PARAMETERS
    # =============================================================================
    info_filename = os.path.join(results_dir, "info.txt")
    with open(info_filename, "w") as f:
        f.write("Simulation Parameters:\n")
        f.write(f"c0 = {c0}\n")
        f.write(f"mu0 = {mu0}\n")
        f.write(f"epsilon0 = {epsilon0}\n")
        f.write(f"eta0 = {eta0}\n")
        f.write(f"wavelength = {wavelength}\n")
        f.write(f"theta_deg = {theta_deg}\n")
        f.write(f"phi_deg = {phi_deg}\n")
        f.write(f"pol_angle_deg = {pol_angle_deg}\n")
        f.write(f"radius = {radius}\n")
        f.write(f"n_real = {n_real}\n")
        f.write(f"n_imag = {n_imag}\n")
        f.write(f"resolution = {resolution}\n")
        f.write(f"npml = {npml}\n")
        f.write(f"maxiters = {maxiters}\n")
        f.write(f"tol = {tol}\n")
        f.write(f"omega = {omega}\n")
        f.write(f"k0 = {k0}\n")
        f.write(f"dx = {dx}, dy = {dy}, dz = {dz}\n")
        f.write(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}\n")
        f.write(f"NPML = {NPML}\n")
        f.write(f"BC = {BC}\n")
        f.write(f"amax = {amax}, cmax = {cmax}, p = {p}\n")
    print("Info file saved to:", info_filename)

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    main()
