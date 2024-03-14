import numpy as np
import numpy.fft as fft
import numba as nb
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.ndimage import convolve, binary_dilation, uniform_filter, gaussian_filter

class Limiter:
    """
    The job of a flux limiter is to prevent oscillations near discontinuities

    They are pretty much the core concept behind TVD schemes, as they prevent the total variation from increasing
    """
    @staticmethod
    def vanLeer(r):
        """
        Van Leer flux limiter

        parameters:
        r : 
            represents the ratio of successive gradients
            r_i= (u_{i}-u_{i-1})/(u_{i+1}-u_{i})
        """
        
        return (r + abs(r)) / (1 + abs(r))
    
    @staticmethod
    def minmod(a, b):
        """
        minmod flux limiter
        """
        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))
    

class Filter:    
    """
    Different filters for separating out small eddies from large ones in LES
    """
    @staticmethod
    def mean(phi:np.ndarray) -> np.ndarray:
        '''Filter NS using convolution kernel to resolve large scales of motion directly while modeling smaller, subgrid-scale motions

        Filters each of pressure, density, and velocity

        Returns resolved portion
        '''
        
        phi_filtered = np.zeros_like(phi)
        for i in range(1, phi.shape[0]-1):
            for j in range(1, phi.shape[1]-1):
                phi_filtered[i, j] = np.mean(phi[i-1:i+1, j-1:j+1])
        return phi_filtered
    
    @staticmethod
    def gaussian(phi:np.ndarray, sigma: float) -> np.ndarray:
        """
        scipy gauss filter - will implement own later
        """
        return gaussian_filter(phi, sigma=sigma)
    

class EquilibriumSolver:
    """
    Different equilibrium solvers to be used for pressure-poisson equation

    Sparse matrices are required for this due to the sheer size of the domain

    Succsessive Over-Relaxation (SOR) Solver
    """
    @staticmethod
    def jacobi(A:csr_matrix, b:np.ndarray, phi:np.ndarray = None,  tolerance:float=1e-6, maxiter:int=100):
        """
        Parameters
        --------------------------------
        `phi` : np.ndarray
            matrix of dimension (n,m)
        `A` : np.ndarray
            Coefficient matrix A of dimension (nm,nm)
        `b` : np.ndarray 
            Source term vector of dimesnion (nm) -> i.e. convective flux 
        `tolerance` : float
            Convergence tolerance

        Returns
        --------------------------------
        Solution to `Ax = b`

        `x^(k+1) = D^-1 (b - (L+U) x^k)`
        """
        shape = b.shape
        a = 0.5
        b = b.flatten()
        if phi is not None:
            x = phi.flatten()
        else:
            x = np.zeros_like(b)
        D_inv = 1.0 / A.diagonal()
        for _ in range(maxiter):
            R = b - A.dot(x)
            x_new = x + a * (D_inv * R)
            if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
                return x_new.reshape(shape)
            x = x_new
        return x_new.reshape(shape)

    @staticmethod
    def gauss_seidel(A:csr_matrix, b:np.ndarray, phi:np.ndarray = None,  tolerance:float=1e-6, maxiter:int=100):
        """
        Parameters
        ----------------
        `phi` : np.ndarray
            Matrix of dimension (n,m)
        `A` : np.ndarray
            Coefficient matrix A of dimension (nm,nm)
        `b` : np.ndarray 
            Source term vector of dimesnion (nm) -> i.e. convective flux 
        `tolerance` : float
            Convergence tolerance

        Returns
        ----------------
        Solution to `Ax = b`

        `Ax = b`    
                kg/(m s^2) -   kg/(m s^2)
        `x_i = 1/(A_ii) * (b_i - Σ_(j≠i) A_ij x_j)`
                            1/s            m^ * kg/m s^2
            - `A_ii` = on-diagonal coefficient (coefficient of x_i) 
            - `A_ij` = off-dsiagonal coefficient            
        """
        shape = b.shape
        b = b.flatten()
        if phi is None:
            x = np.zeros_like(b)
        else:
            x = phi.flatten()
        D_inv = 1.0 / A.diagonal()
        for _ in range(maxiter):
            x_new = x
            for i in range(len(x)):
                Ax_i = A[i, :].dot(x) - A[i, i] * x[i]
                x_new[i] = (b[i] - Ax_i) * D_inv[i]
            if np.linalg.norm(x_new - x, ord=np.inf) / (np.linalg.norm(x_new, ord=np.inf) +1e-9)< tolerance:
                return x_new.reshape(shape)
        return x_new.reshape(shape)

    @staticmethod
    def conjugate_gradient(A:csr_matrix, b:np.ndarray, phi:np.ndarray = None,  tolerance:float=1e-6, maxiter:int=100):
        shape = b.shape
        b = b.flatten()
        if phi is None:
            x = np.zeros_like(b)
        else:
            x = phi.flatten()
        x_new = cg(A, b, x0=x, tol=tolerance, maxiter=maxiter)[0]
        return x_new.reshape(shape)
    



def make_sparse_A(Nx, Ny, dx, dy):
    """
    Creates a sparse coefficient matrix for the discretized pressure-poisson eqaution

    Parameters
    --------------------------------
    `Nx` : int
        Number of cells in x
    `Ny` : int
        Number of cells in y
    `dx` : float
        Grid spacing in x
    `dy` : float
        Grid spacing in y

    Returns
    --------------------------------
    Sparse matrix
    """
    A = lil_matrix((Nx * Ny, Nx * Ny))
    for j in range(Ny):
        for i in range(Nx):
            index = j * Nx + i
            A[index, index] = -2/dx**2 - 2/dy**2

            if i == 0 or i == Nx-1: 
                A[index, index] += 1/dx**2  
                
            if j == 0 or j == Ny-1:  
                A[index, index] += 1/dy**2 
            
            if i > 0:
                A[index, index-1] = 1/dx**2
            if i < Nx-1:
                A[index, index+1] = 1/dx**2

            if j > 0:
                A[index, index-Nx] = 1/dy**2
            if j < Ny-1:
                A[index, index+Nx] = 1/dy**2

    return csr_matrix(A.tocsr())

def inside_cylinder(x, y, radius, center=(0, 0)):
    """
    Creates stencil for inside a cylinder
    """
    return (x - center[0])**2 + (y - center[1])**2 <= radius**2

def adjust_velocity_ghost_cells(u:np.ndarray, nghosts:np.ndarray, dx:float, axis:int=0, flow_profile:str='linear'):
    """
    Adjusts ghost cell velocities to reflect the expected velocity gradient near walls.
    
    Parameters:
    --------------------------------
    `u` : np.ndarray
        
    `nghosts` : int
        Number of ghost cells to consider
    `dx` : flaot
        Grid spacing in the y-direction.
    `axis` : int
    `flow_profile` : str
    """
    if flow_profile == 'linear':
        if axis == 0:
            gradient = (u[-(nghosts+1)] - u[-(nghosts+2)]) / dx
            for i in range(1, nghosts + 1):
                u[(-nghosts-1) + i] = u[-nghosts-1] + gradient * i * dx
        elif axis == 1:
            gradient = (u[:, -(nghosts+1)] - u[:, -(nghosts+2)]) / dx
            for i in range(1, nghosts + 1):
                u[:, (-nghosts-1) + i] = u[:, -nghosts-1] + gradient * i * dx

    elif flow_profile == 'turbulent':
        pass
    return u



class Slices:
    @staticmethod
    def _get_slice_intersect(slice1:slice, slice2:slice, maxval:int) -> slice:
        """

        Parameters
        --------------------------------
        slice1 : slice
        
        slice2 : slice

        maxval : int

        Returns
        --------------------------------
        Intersect of slices in 1D
        """
        start1 = 0 if slice1.start is None else slice1.start
        if start1 < 0: start1 = maxval + start1
        start2 = 0 if slice2.start is None else slice2.start
        if start2 < 0: start2 = maxval + start2
        stop1 = maxval if slice1.stop is None else slice1.stop
        if stop1 < 0: stop1 = maxval + stop1
        stop2 = maxval if slice2.stop is None else slice2.stop
        if stop2 < 0: stop2 = maxval + stop2
        start = max(start1, start2)
        stop = min(stop1, stop2)
        if start >= stop:
            return None
        return slice(start, stop)
    
    @staticmethod
    def get_slice_intersect(slice1:slice | tuple, slice2: slice | tuple, grid_size:tuple) -> slice:
        """
        Note: assumes that tuple is in order (x,y) for 2D

        Parameters
        --------------------------------
        `slice1` : slice | tuple
        
        `slice2` : slice | tuple
        
        Returns
        --------------------------------
        Intersect of slices in 1D or 2D
        """
        if isinstance(slice1, slice) and isinstance(slice2, slice):
            return Slices._get_slice_intersect(slice1, slice2, grid_size)
        elif isinstance(slice1, tuple) and isinstance(slice2, tuple):    
            if len(slice1) != len(slice2):
                raise IndexError(f"argument of length {len(slice1)} cannot be intersected with argument of length {len(slice2)}")
            slice_y1, slice_x1 = slice1
            slice_y2, slice_x2 = slice2
            max_y, max_x = grid_size
            intersect_slice_x =  Slices._get_slice_intersect(slice_x1, slice_x2, max_x)
            intersect_slice_y =  Slices._get_slice_intersect(slice_y1, slice_y2, max_y)
            return (intersect_slice_y, intersect_slice_x)
        else:
            raise TypeError("Both arguments must be slices or tuples of slices")

    @staticmethod
    def _convert_slice(s:slice, max_val:int):
        """
        Parameters
        --------------------------------
        `s` : slice

        `max_val` : int

        Returns
        --------------------------------
        New start and stop of converted slice
        """
        start = 0 if s.start is None else (s.start if s.start >= 0 else max_val + s.start)
        stop = max_val if s.stop is None else (s.stop if s.stop >= 0 else max_val + s.stop)
        return start, stop

    @staticmethod
    def adjust_indices(indices:tuple[slice,slice], intersection:tuple[slice,slice], grid:tuple, corner:str):
        """"
        Note: assumes that tuple is in order (x,y) for 2D

        Parameters
        --------------------------------
        `indices` : tuple[slice,slice]

        `intersection` : tuple[slice,slice]

        `grid` : tuple

        `corner` : str

        Returns
        --------------------------------
        New indices
        """
        if intersection[0] is None or intersection[1] is None: 
            return indices
        max_y, max_x = grid
        y_indices, x_indices = indices
        y_intersect, x_intersect = intersection
        start_y, stop_y = Slices._convert_slice(y_indices, max_y)
        start_x, stop_x = Slices._convert_slice(x_indices, max_x)
        start_y_int, stop_y_int = Slices._convert_slice(y_intersect, max_y)
        start_x_int, stop_x_int = Slices._convert_slice(x_intersect, max_x)
        if corner[0] == 'top' or corner[0] == 'bottom':
            if corner[1] == 'left':
                start_x += (stop_x_int - start_x_int)
            elif corner[1] == 'right':
                stop_x -= (stop_x_int - start_x_int)

        elif corner[0] == 'right' or corner[0] == 'left':
            if corner[1] == 'bottom':
                start_y += (stop_y_int - start_y_int)
            elif corner[1] == 'top':
                stop_y -= (stop_y_int - start_y_int)

        new_x = slice(start_x, stop_x) if start_x < stop_x else slice(0,0)
        new_y = slice(start_y, stop_y) if start_y < stop_y else slice(0,0)
        return new_y, new_x