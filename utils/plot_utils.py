# utils/plot_utils.py

import os
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc
import numpy as np

import config
# plt.style.use('dark_background')
# plt.rcParams['figure.figsize'] = (15, 15)
# plt.rcParams['axes.facecolor'] = '#000000'
# plt.rcParams['grid.color'] = 'gray'
# plt.rcParams['grid.alpha'] = 0.2
# plt.rcParams['axes.labelcolor'] = 'white'
# plt.rcParams['xtick.color'] = 'white'
# plt.rcParams['ytick.color'] = 'white'
font_size = 14
#larger font
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'axes.labelsize': font_size})
plt.rcParams.update({'xtick.labelsize': font_size})
plt.rcParams.update({'ytick.labelsize': font_size})
plt.rcParams.update({'legend.fontsize': font_size})

def plot_slices_3d(field3d, xvals, yvals, zvals, 
                   cmap='cet_CET_D4_r', title="", vmin=None, vmax=None):
    """
    Plots real(field3d) slices in 3 orthogonal planes:
      1) z=0 plane
      2) y=0 plane
      3) x=0 plane
    This helps visualize scattering/refraction in multiple cross-sections.
    field3d shape = (Nx, Ny, Nz).
    """
    # Indices near 0-plane
    ix0 = np.argmin(np.abs(xvals))
    iy0 = np.argmin(np.abs(yvals))
    iz0 = np.argmin(np.abs(zvals))

    # Slices
    # XY-plane @ z=0
    slice_xy = np.real(field3d[:, :, iz0]).T
    # XZ-plane @ y=0
    slice_xz = np.real(field3d[:, iy0, :]).T
    # YZ-plane @ x=0
    slice_yz = np.real(field3d[ix0, :, :]).T

    # Setup 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axs[0].imshow(slice_xy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]])
    axs[0].set_title(f"{title}\nZ=0 slice", fontsize=10)
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(slice_xy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[xvals[0], xvals[-1], zvals[0], zvals[-1]])
    axs[1].set_title("Y=0 slice", fontsize=10)
    axs[1].set_xlabel('x (m)')
    axs[1].set_ylabel('z (m)')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(slice_xy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[yvals[0], yvals[-1], zvals[0], zvals[-1]])
    axs[2].set_title("X=0 slice", fontsize=10)
    axs[2].set_xlabel('y (m)')
    axs[2].set_ylabel('z (m)')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    save_path = config.SAVE_DIR + title + ".png"
    plt.savefig(save_path, dpi=300)
    
def scatter_3d_field(X, Y, Z, values, stride=1, title="", cmap="viridis", alpha=0.8):
    """
    Produce a 3D scatter plot of the given field.
    Only a subset of points is plotted (by applying 'stride') to improve clarity.
    """
    from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    Nz = Z.shape[2]
    z_start = Nz // 3
    z_end = 2 * Nz // 3
    X_ = X[::stride, ::stride, z_start:z_end:stride].ravel()
    Y_ = Y[::stride, ::stride, z_start:z_end:stride].ravel()
    Z_ = Z[::stride, ::stride, z_start:z_end:stride].ravel()
    vals_ = values[::stride, ::stride, z_start:z_end:stride].ravel()
    p = ax.scatter(X_, Y_, Z_, c=vals_, s=2, cmap=cmap, alpha=alpha)
    cbar = fig.colorbar(p, ax=ax, fraction=0.03, pad=0.07)
    cbar.set_label("Amplitude")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    plt.tight_layout()
    save_path = config.SAVE_DIR + title + ".png"
    plt.savefig(save_path, dpi=300)

    
def auto_vlims(*arrays):
    """
    Returns (vmin, vmax) based on the maximum absolute real part
    among the given numpy arrays.
    """
    maxval = 0.0
    for arr in arrays:
        maxval = max(maxval, np.max(np.abs(np.real(arr))))
    return (-maxval, maxval)


def plot_pml_slices(sx, sy, sz, title="pml_slices"):
    """
    Plot slices of the 3D arrays sx, sy, sz in three orthogonal planes:
      - XY-plane at z=0
      - YZ-plane at x=0
      - XZ-plane at y=0

    Each array is shape (Nx, Ny, Nz), and we use imshow() to display
    magnitude |sx|, |sy|, |sz| on those planes.
    """

    Nx, Ny, Nz = sx.shape

    # Gather slices for each array
    # For the XY-plane at z=0:
    sx_xy = np.abs(sx[:, :, 0])
    sy_xy = np.abs(sy[:, :, 0])
    sz_xy = np.abs(sz[:, :, 0])

    # For the YZ-plane at x=0:
    sx_yz = np.abs(sx[0, :, :])   # shape (Ny, Nz)
    sy_yz = np.abs(sy[0, :, :])
    sz_yz = np.abs(sz[0, :, :])

    # For the XZ-plane at y=0:
    sx_xz = np.abs(sx[:, 0, :])   # shape (Nx, Nz)
    sy_xz = np.abs(sy[:, 0, :])
    sz_xz = np.abs(sz[:, 0, :])

    # Create a 3x3 figure: rows = s_x, s_y, s_z; columns = XY, YZ, XZ
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    # Row 0 => sx, Row 1 => sy, Row 2 => sz

    # --- Row 0: s_x slices ---
    # XY-plane
    ax = axes[0, 0]
    im = ax.imshow(sx_xy.T, origin='lower', aspect='auto')
    ax.set_title("|sx| in XY (z=0)")
    plt.colorbar(im, ax=ax)

    # YZ-plane
    ax = axes[0, 1]
    im = ax.imshow(sx_yz.T, origin='lower', aspect='auto')
    ax.set_title("|sx| in YZ (x=0)")
    plt.colorbar(im, ax=ax)

    # XZ-plane
    ax = axes[0, 2]
    im = ax.imshow(sx_xz.T, origin='lower', aspect='auto')
    ax.set_title("|sx| in XZ (y=0)")
    plt.colorbar(im, ax=ax)

    # --- Row 1: s_y slices ---
    # XY-plane
    ax = axes[1, 0]
    im = ax.imshow(sy_xy.T, origin='lower', aspect='auto')
    ax.set_title("|sy| in XY (z=0)")
    plt.colorbar(im, ax=ax)

    # YZ-plane
    ax = axes[1, 1]
    im = ax.imshow(sy_yz.T, origin='lower', aspect='auto')
    ax.set_title("|sy| in YZ (x=0)")
    plt.colorbar(im, ax=ax)

    # XZ-plane
    ax = axes[1, 2]
    im = ax.imshow(sy_xz.T, origin='lower', aspect='auto')
    ax.set_title("|sy| in XZ (y=0)")
    plt.colorbar(im, ax=ax)

    # --- Row 2: s_z slices ---
    # XY-plane
    ax = axes[2, 0]
    im = ax.imshow(sz_xy.T, origin='lower', aspect='auto')
    ax.set_title("|sz| in XY (z=0)")
    plt.colorbar(im, ax=ax)

    # YZ-plane
    ax = axes[2, 1]
    im = ax.imshow(sz_yz.T, origin='lower', aspect='auto')
    ax.set_title("|sz| in YZ (x=0)")
    plt.colorbar(im, ax=ax)

    # XZ-plane
    ax = axes[2, 2]
    im = ax.imshow(sz_xz.T, origin='lower', aspect='auto')
    ax.set_title("|sz| in XZ (y=0)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = config.SAVE_DIR + title + ".png"
    plt.savefig(save_path, dpi=300)


from matplotlib.patches import Circle

def plot_slices_with_pml(
    field3d, xvals, yvals, zvals,
    pml_x_lo, pml_x_hi, pml_y_lo, pml_y_hi, pml_z_lo, pml_z_hi,
    cmap='seismic_r', title="", vmin=None, vmax=None,
    x_slice=None, y_slice=None, z_slice=None,
    pml_geometry='cartesian', BC=(0,0,0),
    alpha=1.0, font_size=12
):
    """
    Plot three orthogonal slices (XY, XZ, YZ) of a 3D field with markers showing the PML boundaries.
    
    The drawn boundaries update according to the chosen PML geometry and the boundary conditions.
    
    Inputs:
      - field3d: 3D numpy array.
      - xvals, yvals, zvals: 1D coordinate arrays.
      - pml_x_lo, pml_x_hi, etc.: PML thickness values (in number of cells).
      - cmap, title, vmin, vmax, x_slice, y_slice, z_slice: plotting options.
      - pml_geometry: string, one of 'cartesian', 'cylindrical', or 'spherical'.
      - BC: a tuple (xbc, ybc, zbc); if BC[i]==1 then that direction is periodic and no boundary is drawn.
      - alpha: transparency for the imshow plots.
      - font_size: font size for labels and titles.
    """
    # Compute common color limits.
    vmin, vmax = auto_vlims(field3d)
    
    def find_slice_index(vals, target=None):
        if target is None:
            return np.argmin(np.abs(vals))
        return np.argmin(np.abs(vals - target))
    
    ix0 = find_slice_index(xvals, x_slice)
    iy0 = find_slice_index(yvals, y_slice)
    iz0 = find_slice_index(zvals, z_slice)
    
    # Extract slices.
    slice_xy = np.real(field3d[:, :, iz0]).T
    slice_xz = np.real(field3d[:, iy0, :]).T
    slice_yz = np.real(field3d[ix0, :, :]).T

    fig, axs = plt.subplots(1, 3, figsize=(36, 12))
    
    # ---------------------------
    # For the x-y slice:
    # ---------------------------
    ax_xy = axs[0]
    im0 = ax_xy.imshow(slice_xy, origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax, alpha=alpha,
                       extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]])
    ax_xy.set_title(f"{title}\nZ = {zvals[iz0]:.2e}", fontsize=font_size)
    ax_xy.set_xlabel("x (m)", fontsize=font_size)
    ax_xy.set_ylabel("y (m)", fontsize=font_size)
    plt.colorbar(im0, ax=ax_xy, fraction=0.046, pad=0.04)
    
    # Draw PML boundaries in x-y:
    if pml_geometry.lower() == 'cartesian':
        # Draw vertical/horizontal dashed lines if the corresponding BC==0.
        if BC[0] == 0:
            ax_xy.axvline(xvals[pml_x_lo], color='k', linestyle='--', alpha=alpha)
            ax_xy.axvline(xvals[-(pml_x_hi+1)], color='k', linestyle='--', alpha=alpha)
        if BC[1] == 0:
            ax_xy.axhline(yvals[pml_y_lo], color='k', linestyle='--', alpha=alpha)
            ax_xy.axhline(yvals[-(pml_y_hi+1)], color='k', linestyle='--', alpha=alpha)
    elif pml_geometry.lower() in ['cylindrical', 'spherical']:
        # In the x-y slice, draw a circle if either x or y is Dirichlet.
        if (BC[0] == 0) or (BC[1] == 0):
            # Compute center of the x-y domain.
            center_x = (xvals[0] + xvals[-1]) / 2.0
            center_y = (yvals[0] + yvals[-1]) / 2.0
            # For the effective boundary, if BC==0 in x then use the corresponding value;
            # otherwise use the domain edge.
            left = xvals[pml_x_lo] if BC[0] == 0 else xvals[0]
            right = xvals[-(pml_x_hi+1)] if BC[0] == 0 else xvals[-1]
            bottom = yvals[pml_y_lo] if BC[1] == 0 else yvals[0]
            top = yvals[-(pml_y_hi+1)] if BC[1] == 0 else yvals[-1]
            # Use the minimum distance from the center.
            r_boundary = min(center_x - left, right - center_x, center_y - bottom, top - center_y)
            circ = Circle((center_x, center_y), r_boundary,
                          edgecolor='k', facecolor='none', linestyle='--', linewidth=2, alpha=alpha)
            ax_xy.add_patch(circ)
    # ---------------------------
    # For the x-z slice:
    # ---------------------------
    ax_xz = axs[1]
    im1 = ax_xz.imshow(slice_xz, origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax, alpha=alpha,
                       extent=[xvals[0], xvals[-1], zvals[0], zvals[-1]])
    ax_xz.set_title(f"Y = {yvals[iy0]:.2e}", fontsize=font_size)
    ax_xz.set_xlabel("x (m)", fontsize=font_size)
    ax_xz.set_ylabel("z (m)", fontsize=font_size)
    plt.colorbar(im1, ax=ax_xz, fraction=0.046, pad=0.04)
    # For x-z slice, if x is Dirichlet, draw vertical lines; if z is Dirichlet, horizontal.
    if BC[0] == 0:
        ax_xz.axvline(xvals[pml_x_lo], color='k', linestyle='--', alpha=alpha)
        ax_xz.axvline(xvals[-(pml_x_hi+1)], color='k', linestyle='--', alpha=alpha)
    if BC[2] == 0:
        ax_xz.axhline(zvals[pml_z_lo], color='k', linestyle='--', alpha=alpha)
        ax_xz.axhline(zvals[-(pml_z_hi+1)], color='k', linestyle='--', alpha=alpha)
    
    # ---------------------------
    # For the y-z slice:
    # ---------------------------
    ax_yz = axs[2]
    im2 = ax_yz.imshow(slice_yz, origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax, alpha=alpha,
                       extent=[yvals[0], yvals[-1], zvals[0], zvals[-1]])
    ax_yz.set_title(f"X = {xvals[ix0]:.2e}", fontsize=font_size)
    ax_yz.set_xlabel("y (m)", fontsize=font_size)
    ax_yz.set_ylabel("z (m)", fontsize=font_size)
    plt.colorbar(im2, ax=ax_yz, fraction=0.046, pad=0.04)
    if BC[1] == 0:
        ax_yz.axvline(yvals[pml_y_lo], color='k', linestyle='--', alpha=alpha)
        ax_yz.axvline(yvals[-(pml_y_hi+1)], color='k', linestyle='--', alpha=alpha)
    if BC[2] == 0:
        ax_yz.axhline(zvals[pml_z_lo], color='k', linestyle='--', alpha=alpha)
        ax_yz.axhline(zvals[-(pml_z_hi+1)], color='k', linestyle='--', alpha=alpha)
    
    fig.suptitle(title, fontsize=font_size)
    plt.tight_layout()
    save_path = os.path.join(config.SAVE_DIR, title + ".png")
    plt.savefig(save_path, dpi=300)

def plot_epsilon(eps, xvals, yvals, zvals):
    """
    Plot three orthogonal slices of the (real part of) epsilon.
    """
    Nx, Ny, Nz = eps.shape
    slice_index = Nz // 2

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axs[0].imshow(np.real(eps[:, :, slice_index]).T, origin='lower',
                        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
                        cmap='viridis')
    axs[0].set_title("Epsilon slice at z = 0")
    plt.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(np.real(eps[:, Ny//2, :]).T, origin='lower',
                        extent=[xvals[0], xvals[-1], zvals[0], zvals[-1]],
                        cmap='viridis')
    axs[1].set_title("Epsilon slice at y = 0")
    plt.colorbar(im1, ax=axs[1])
    
    im2 = axs[2].imshow(np.real(eps[Nx//2, :, :]).T, origin='lower',
                        extent=[yvals[0], yvals[-1], zvals[0], zvals[-1]],
                        cmap='viridis')
    axs[2].set_title("Epsilon slice at x = 0")
    plt.colorbar(im2, ax=axs[2])
    
    fig.suptitle("Permittivity Slices (Real Part)")
    plt.tight_layout()
    save_path = config.SAVE_DIR + "epsilon_slices.png"
    plt.savefig(save_path, dpi=300)