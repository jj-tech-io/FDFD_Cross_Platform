import numpy as np
import numpy as np
import scipy.sparse as sp

def yeeder3d(NS, RES, BC, kinc=None):
    """
    YEEDER3D - Constructs 3D derivative matrices on a Yee grid (flattened in Fortran order),
    with quasi‑periodic (Bloch) boundary conditions implemented in the derivative operator.
    
    For a periodic direction (BC[d]==1), the finite-difference operator is wrapped using
    the phase factor exp(-1j * k_d * (N_d * d_d)), so that for example in z:
    
        E(x,y,z+L) = exp(-1j * kz * L) * E(x,y,z)
    
    Inputs:
      NS   : (Nx, Ny, Nz) grid dimensions.
      RES  : (dx, dy, dz) cell sizes.
      BC   : (xbc, ybc, zbc) boundary conditions (0 => Dirichlet; 1 => Periodic).
      kinc : (kx, ky, kz) incident wavevector (default = [1, 0, 0]).
             Used to compute the wrap-phase in the periodic directions.
    
    Outputs:
      DEX, DEY, DEZ : Sparse forward-difference derivative matrices for E-fields.
      DHX, DHY, DHZ : Sparse derivative matrices for H-fields (=-transpose(DEX), etc.).
    """
    Nx, Ny, Nz = NS
    dx, dy, dz = RES
    if kinc is None:
        kinc = [1.0, 0.0, 0.0]
    kinc = np.array(kinc, dtype=np.complex128)
    kx, ky, kz = kinc

    M = Nx * Ny * Nz

    # --- DEX (x-derivative) ---
    if Nx == 1:
        DEX = -1j * kx * sp.eye(M, format='csc')
    else:
        # Forward difference: (f[i+1] - f[i])/dx
        d0 = -np.ones(M)
        d1 = np.ones(M)
        # Prevent index wrapping for Dirichlet boundary:
        d1[np.arange(Nx-1, M, Nx)] = 0
        data = np.vstack((d0, d1)) / dx
        DEX = sp.diags(data, [0, 1], shape=(M, M), format='csc')
        if BC[0] == 1:
            # Wrap from i=Nx-1 to i=0, with phase factor exp(-1j*kx*Nx*dx)
            wrap_data = np.zeros(M, dtype=np.complex128)
            wrap_data[0:M:Nx] = np.exp(-1j * kx * (Nx*dx)) / dx
            DEX += sp.diags([wrap_data], [1-Nx], shape=(M, M))

    # --- DEY (y-derivative) ---
    if Ny == 1:
        DEY = -1j * ky * sp.eye(M, format='csc')
    else:
        d0 = -np.ones(M, dtype=np.complex128)
        block = np.concatenate((np.ones((Ny-1)*Nx, dtype=np.complex128),
                                np.zeros(Nx, dtype=np.complex128)))
        d1 = np.tile(block, Nz)
        data = np.vstack((d0, d1)) / dy
        DEY = sp.diags(data, [0, Nx], shape=(M, M), format='csc')
        if BC[1] == 1:
            ph = np.exp(-1j * ky * (Ny*dy)) / dy
            wrap_block = np.concatenate((np.ones(Nx, dtype=np.complex128),
                                          np.zeros((Ny-1)*Nx, dtype=np.complex128)))
            wrap_data = np.tile(wrap_block, Nz) * ph
            DEY += sp.diags([wrap_data], [Nx*(1-Ny)], shape=(M, M))
    # --- DEZ (z-derivative) ---
    if Nz == 1:
        DEZ = -1j * kz * sp.eye(M, format='csc')
    else:
        # Forward difference: (f[k+1] - f[k])/dz
        d0 = -np.ones(M, dtype=np.complex128)
        d1 =  np.ones(M,  dtype=np.complex128)

        # For non-periodic z, no forward difference from the top z-slice:
        # The last Nx*Ny entries are the top z-slice in Fortran ordering:
        if BC[2] == 0:
            d1[-(Nx*Ny):] = 0

        data = np.vstack((d0, d1)) / dz
        DEZ  = sp.diags(data, [0, Nx*Ny], shape=(M, M), format='csc')

        if BC[2] == 1:
            # Periodic wrap-around in z
            wrap_vals = np.exp(-1j * kz * (Nz*dz)) / dz
            wrap_data = wrap_vals * np.ones(Nx*Ny, dtype=np.complex128)
            DEZ += sp.diags([wrap_data], [-Nx*Ny*(Nz-1)], shape=(M, M), format='csc')

    # --- H-field operators (negative transpose) ---
    DHX = -DEX.transpose(copy=True)
    DHY = -DEY.transpose(copy=True)
    DHZ = -DEZ.transpose(copy=True)

    return DEX, DEY, DEZ, DHX, DHY, DHZ


def yeeder2d(NS, RES, BC, kinc=None):
    """
    YEEDER2D - Constructs 2D derivative matrices on a Yee grid (flattened in Fortran order),
    with quasi‑periodic (Bloch) boundary conditions implemented in the derivative operator.

    Inputs:
        NS : (Nx, Ny) grid dimensions.
        RES: (dx, dy) cell sizes.
        BC : (xbc, ybc) boundary conditions (0 => Dirichlet; 1 => Periodic).
        kinc: (kx, ky) incident wavevector (default = [1, 0]).
              Used to compute the wrap-phase in the periodic directions.

    Outputs:
        DEX, DEY : Sparse forward-difference derivative matrices for E-fields.
        DHX, DHY : Sparse derivative matrices for H-fields (=-transpose(DEX), etc.).
    """
    Nx, Ny = NS
    dx, dy = RES
    if kinc is None:
        kinc = [1.0, 0.0]
    kinc = np.array(kinc, dtype=np.complex128)
    kx, ky = kinc

    M = Nx * Ny

    # --- DEX (x-derivative) ---
    if Nx == 1:
        DEX = -1j * kx * sp.eye(M, format='csc')
    else:
        # Forward difference in x: (f[i+1] - f[i]) / dx
        d0 = -np.ones(M)
        d1 = np.ones(M)
        # Prevent wrapping on the right edge for Dirichlet BC:
        d1[np.arange(Nx-1, M, Nx)] = 0
        data = np.vstack((d0, d1)) / dx
        DEX = sp.diags(data, [0, 1], shape=(M, M), format='csc')
        if BC[0] == 1:
            # Periodic wrap: add values from the last column back to the first column
            wrap_data = np.zeros(M, dtype=np.complex128)
            wrap_data[0:M:Nx] = np.exp(-1j * kx * (Nx*dx)) / dx
            DEX += sp.diags([wrap_data], [1-Nx], shape=(M, M))

    # --- DEY (y-derivative) ---
    if Ny == 1:
        DEY = -1j * ky * sp.eye(M, format='csc')
    else:
        # Forward difference in y: (f[i+Nx] - f[i]) / dy
        d0 = -np.ones(M, dtype=np.complex128)
        # Create a block that has ones for all rows except the last row of each column
        block = np.concatenate((np.ones((Ny-1)*Nx, dtype=np.complex128),
                                np.zeros(Nx, dtype=np.complex128)))
        d1 = block
        data = np.vstack((d0, d1)) / dy
        DEY = sp.diags(data, [0, Nx], shape=(M, M), format='csc')
        if BC[1] == 1:
            ph = np.exp(-1j * ky * (Ny*dy)) / dy
            wrap_block = np.concatenate((np.ones(Nx, dtype=np.complex128),
                                         np.zeros((Ny-1)*Nx, dtype=np.complex128)))
            wrap_data = wrap_block * ph
            DEY += sp.diags([wrap_data], [Nx*(1-Ny)], shape=(M, M))
    
    # --- H-field derivative operators (negative transpose) ---
    DHX = -DEX.transpose(copy=True)
    DHY = -DEY.transpose(copy=True)

    return DEX, DEY, DHX, DHY


#main entry point
if __name__ == "__main__":
    #test yeeder 2d
    """
    9   NS  = [3 4];
    10  RES = [0.2 0.1];
    11  BC  = [0 0];
    """
    NS = [3, 4]
    RES = [0.2, 0.1]
    BC = [0, 0]
    DEX, DEY, DHX, DHY = yeeder2d(NS, RES, BC)
    print(DEX.toarray())
    print(DEY.toarray())
    print(DHX.toarray())
    print(DHY.toarray())
    #test yeeder 3d
    """
     9   NS  = [2 2 2];
    10  RES = [0.3 0.2 0.1];
    11  BC  = [0 0 0];
    """
    NS = [2, 2, 2]
    RES = [0.3, 0.2, 0.1]
    BC = [0, 0, 0]
    DEX, DEY, DEZ, DHX, DHY, DHZ = yeeder3d(NS, RES, BC)
    for mat in [DEX, DEY, DEZ, DHX, DHY, DHZ]:
        print(mat.toarray())