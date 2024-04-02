from .utils import *
import numpy as np
from numba import njit

__all__ = ["first_order_partial", 
           "second_order_partial", 
           "laplacian",
           "gradient",
           "divergence",
           "curl",
           "interpolate_velocity_to_faces"]


@njit
def first_order_partial(phi:np.ndarray, dx:float, axis:int, method:str='linear') -> np.ndarray:
    """
    First order partial helper using central difference method

    Parameters
    --------------------------------
    `phi` : np.ndarray 
    
    `dx` : float
     
    `axis` : int 
    
    `method` : str

    Returns
    --------------------------------
    """

    dphi_dx = np.zeros_like(phi)
    if axis == 0:
        # Central
        dphi_dx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / 2
        # Forward
        dphi_dx[0,    :] = (phi[1,  :] - phi[0,   :])
        # Backward
        dphi_dx[-1,   :] = (phi[-1, :] - phi[-2,  :])
    elif axis == 1:
        #phi = np.pad(phi, ((0,0),(1,1)), "edge")
        dphi_dx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / 2
        dphi_dx[:,    0] = (phi[:,  1] - phi[:,   0])
        dphi_dx[:,   -1] = (phi[:, -1] - phi[:,  -2])
    return dphi_dx / dx


def second_order_partial(phi:np.ndarray, dx:float, axis:int) -> np.ndarray:
    """
    Second order partial helper using central difference method

    Parameters
    --------------------------------
    `phi` : np.ndarray 
    
    `dx` : float
     
    `axis` : int 

    Returns
    --------------------------------
    """

    ddphi_ddx = np.zeros_like(phi)
    
    if axis == 0:        
        ddphi_ddx = first_order_partial(first_order_partial(phi, dx, axis=0), dx, axis=0)
        dnump = np.gradient(np.gradient(phi, dx, axis=0), dx, axis=0)

    elif axis == 1:
        ddphi_ddx = first_order_partial(first_order_partial(phi, dx, axis=1), dx, axis=1)
        dnump = np.gradient(np.gradient(phi, dx, axis=1), dx, axis=1)
    
    #np.testing.assert_almost_equal(dnump, ddphi_ddx, err_msg="ARRGRGAGRG")
    return ddphi_ddx # / (dx ** 2)



def laplacian(phi:np.ndarray, dx:float, dy:float) -> np.ndarray:
    """
    Laplace operater

    Parameters
    --------------------------------
    `phi` : np.ndarray 
    
    `dx` : float
     
    `dy` : float 

    Returns
    --------------------------------

    ∇^2(phi) = 

    (f(i+1, j) + f(i-1, j) - 2 * f(i,j)) / dx^2 +
    (f(i, j+1) + f(i, j-1) - 2 * f(i,j)) / dy^2

    (f(i+1, j) + f(i-1, j) - 2 * f(i,j)) * dy^2 + (f(i, j+1) + f(i, j-1) - 2 * f(i,j)) * dx ^2 / dx^2 * dy^2

    """

    return second_order_partial(phi, dx, axis=1) + second_order_partial(phi, dy, axis=0)

# @nb.njit
def gradient(phi:np.ndarray, dx:float, axis:int) -> np.ndarray:
    """
    Gradient

    Parameters
    --------------------------------
    `phi` : np.ndarray 
    
    `dx` : float
     
    `axis` : int 

    Returns
    --------------------------------

    ∇ * phi
    """
    return np.gradient(phi, dx, axis)



def divergence(phi_x:np.ndarray, phi_y:np.ndarray, dx:float, dy:float) -> np.ndarray:
    """
    Divergence operator

    Parameters
    --------------------------------
    `phi_x` : np.ndarray 

    `phi_y` : np.ndarray 
    
    `dx` : float
     
    `dy` : float 

    Returns
    --------------------------------

    ∇·phi
    """
    div = first_order_partial(phi_x, dx, axis=1) + first_order_partial(phi_y, dy, axis=0)
    #div = gradient(phi_x, dx, axis=1) + gradient(phi_y, dy, axis=0)
    return div

def curl(phi_x:np.ndarray, phi_y:np.ndarray, dx:float, dy:float) -> np.ndarray:
    """
    Curl

    Parameters
    --------------------------------
    `phi_x` : np.ndarray 

    `phi_y` : np.ndarray 
    
    `dx` : float
     
    `dy` : float 
    
    Returns
    --------------------------------

    ∇ x phi
    """
    return first_order_partial(phi_y, dx, axis=1) - first_order_partial(phi_x, dy, axis=0)

# TODO Fix or remove
def interpolate_velocity_to_faces(u_cell:np.ndarray, inlet_velocity:float) -> tuple[np.ndarray]:
    """
    Parameters
    --------------------------------
    u_cell : np.ndarray

    inlet_velocity : float
    
    Returns
    --------------------------------
    """
    
    # TODO change this depending on boundary condition
    # Neumann Boundary Condition
    u_cell = np.pad(u_cell, ((0, 0), (1, 1)), 'edge')
    # Dirichlet Boundaary Condition
    u_cell = np.pad(u_cell, ((0, 1), (0, 0)), 'constant', constant_values=inlet_velocity)
    u_cell = np.pad(u_cell, ((1, 0), (0, 0)), 'constant', constant_values=0)
    u_e = (u_cell[1:-1, 2: ] + u_cell[1:-1, 1:-1]) / 2
    u_w = (u_cell[1:-1, :-2] + u_cell[1:-1, 1:-1]) / 2        
    u_n = (u_cell[2:,  1:-1] + u_cell[1:-1, 1:-1]) / 2
    u_s = (u_cell[:-2, 1:-1] + u_cell[1:-1, 1:-1]) / 2
    return (u_e, u_w, u_n, u_s)

