import numpy as np
from PDE import *
from mesh import *
from plotting import *

class LES:
    """
    
    """
    def __init__(self, size:tuple, ncells:tuple, BCS:dict, constants:dict, nghosts:int=1):

        """
        Initializes the CFDSolve class and creates an instance of the Mesh class using the input parameters

        Parameters
        --------------------------------
        `size` : tuple
            (x, y) size of domain
        `ncells` : tuple
            (nX, nY) number of grid cells in each direction
        `BCS`: dict
            Dictionary of boundary conditions
        `constants` : dict

        `nghosts` : int = 1 

        Also prints fun logo heheh
        """
        self.mesh = Mesh(size, ncells, BCS, nghosts, constants)
        print()

    def _initialize_constants(self, **kwargs) -> None:
        """
        initializes kwargs constants
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        pass

    def _strain_rate_tensor2D(self) -> np.ndarray:
        """
        Calculate the magnitude of the strain rate tensor |S| for a 2D velocity field.
        
        deformation of the resolved scales of motion -> must use filtered velocities


        S_ij = 1/2 (du_i/dx_j + du_j/dx_i) ->

        S_ij = |      du_dx            0.5*(du_dy + dv_dx) |
                   | 0.5*(du_dy + dv_dx)         dv_dy         |

        |S| = sqrt(2*(S_11^2 + 2*S_12^2 + S_22^2))

        Returns
        ---------------
        4D array of the strain-rate tensor
        """

        u, v = self.mesh['u'][:,:], self.mesh['v'][:,:]

        du_dx = first_order_partial(u, self.mesh.dx, axis=1)
        du_dy = first_order_partial(u, self.mesh.dx, axis=0)
        dv_dy = first_order_partial(v, self.mesh.dx, axis=0)
        dv_dx = first_order_partial(v, self.mesh.dx, axis=1)
        tensor2D = np.zeros((2,2, u.shape[0], u.shape[1]))
        tensor2D[0, 0] = du_dx
        tensor2D[1, 1] = dv_dy
        tensor2D[1, 0] = 0.5 * (du_dy + dv_dx)
        tensor2D[0, 1] = 0.5 * (du_dy + dv_dx)
        return tensor2D
    
    def _strain_rate_magnitude(self) -> np.ndarray:
        """
        Returns the magnitude of the strain rate tensor
        """
        tensor2D = self._strain_rate_tensor2D()
        s_mag = tensor2D ** 2
        return np.sqrt(2 * (s_mag[0, 0] + s_mag[1, 1] + s_mag[1, 0] + s_mag[0, 1]))
    
    def smagorinsky_viscosity(self, Cs:float=0.15) -> np.ndarray:
        """
        Calculates the Smagorinsky subgrid-scale viscosity for a 2D uniform grid.
        
        Params:
        --------------------------------
        Cs : float
            Smagorinsky constant -> in range of [0.1,0.2]
        
        Returns
        --------------------------------
        nu_T : np.ndarray
            2D array of the subgrid-scale viscosity at each grid cell.
        """
        S_mag = self._strain_rate_magnitude()
        
        # Calculate the filter width Δ
        Delta_width = np.sqrt(self.mesh.dx * self.mesh.dy)
        return (Cs * Delta_width)**2 * S_mag
    
    def _filtered_stress(self):
        """
        Calculates the resolved stress tensor of the filtered velocity field
        """
        tau = 2 * self.mesh.mu * self._strain_rate_tensor2D()
        kron = (2/3) * self.mesh.mu * divergence(self.mesh['u'][:,:], self.mesh['v'][:,:], self.mesh.dx, self.mesh.dy)
        tau[1,1] -= kron
        tau[0,0] -= kron
        return tau
    
    def _sgs_stress(self):
        """
        Calculates the subgrid-scale stress tensor
        """
        nu_sgs = self.smagorinsky_viscosity()
        ros_tensor = self._strain_rate_tensor2D()
        sgs_stress = -2 * nu_sgs * ros_tensor
        return sgs_stress
        
    def calculate_smagorinsky_coefficient(self):
        """
        Calculates the Smagorinsky Coefficient C_s = 
        """
        return

    def quick_scheme(self, phi:str) -> tuple[np.ndarray]:
        """
        QUICK scheme -> Calculates the value of phi at the cell faces

        Parameters
        --------------------------------
        `phi` : scalar to interpolate

        Returns
        --------------------------------
        A tuple of 4 arrays of the interpolated values of phi at the east, west, north, and south faces of each cell
        """
        field = self.mesh[phi]
        phi_field = field[:,:]
        phi_x, phi_y = phi_field, phi_field
        u, v = self.mesh['u'][:,:], self.mesh['v'][:,:]

        pos_e, neg_e, pos_w, neg_w = [np.zeros_like(u) for _ in range(4)]
        pos_n, neg_n, pos_s, neg_s = [np.zeros_like(u) for _ in range(4)]
        pos_e[:, 2:-2] = (6/8) * phi_x[:, 2:-2] + (3/8) * phi_x[:, 3:-1] - (1/8) * phi_x[:, 1:-3]
        neg_e[:, 2:-2] = (6/8) * phi_x[:, 3:-1] + (3/8) * phi_x[:, 2:-2] - (1/8) * phi_x[:, 4:  ]
        #phi_w + phi_p + phi_ww
        pos_w[:, 2:-2] = (6/8) * phi_x[:, 1:-3] + (3/8) * phi_x[:, 2:-2] - (1/8) * phi_x[:,  :-4]
        neg_w[:, 2:-2] = (6/8) * phi_x[:, 2:-2] + (3/8) * phi_x[:, 1:-3] - (1/8) * phi_x[:, 3:-1]
        pos_n[2:-2, :] = (6/8) * phi_y[2:-2, :] + (3/8) * phi_y[3:-1, :] - (1/8) * phi_y[1:-3, :]
        neg_n[2:-2, :] = (6/8) * phi_y[3:-1, :] + (3/8) * phi_y[2:-2, :] - (1/8) * phi_y[4:,   :]
        pos_s[2:-2, :] = (6/8) * phi_y[1:-3, :] + (3/8) * phi_y[2:-2, :] - (1/8) * phi_y[:-4,  :]
        neg_s[2:-2, :] = (6/8) * phi_y[2:-2, :] + (3/8) * phi_y[1:-3, :] - (1/8) * phi_y[3:-1, :]    

        phi_e = np.where(u >= 0, pos_e, neg_e)
        phi_w = np.where(u >= 0, pos_w, neg_w)    
        phi_n = np.where(v >= 0, pos_n, neg_n) 
        phi_s = np.where(v >= 0, pos_s, neg_s)
        # TODO
        """phi_e = np.where(u == 0, phi_field, phi_e)
        phi_w = np.where(u == 0, phi_field, phi_w)
        phi_n = np.where(v == 0, phi_field, phi_n)
        phi_s = np.where(v == 0, phi_field, phi_s)"""
        phi_e, phi_w, phi_n, phi_s = field.apply_boundary_faces(phi_e, phi_w, phi_n, phi_s)

        return (phi_e, phi_w, phi_n, phi_s) 

    def _convective_flux(self, phi:str) -> tuple[np.ndarray]:
        """
        Calculates the momentum convective flux through a face

        Parameters
        --------------------------------
        `phi` : str

        Returns
        --------------------------------
        """
        u, v = self.mesh['u'][:,:], self.mesh['v'][:,:]
        phi_e, phi_w, phi_n, phi_s = self.quick_scheme(phi)

        # convective fluxes for v-veloctity, where v_e,... are interpolated using QUICK scheme
        # for incompressible flows where density is constant, common not to see rho
        flux_e, flux_w = u * phi_e, u * phi_w
        flux_n, flux_s = v * phi_n, v * phi_s
        net_phi_flux_x = flux_w - flux_e
        net_phi_flux_y = flux_s - flux_n
        return (net_phi_flux_x, net_phi_flux_y)
    

    def _viscous_flux(self) -> tuple[np.ndarray]:
        """
        calculates the viscous flux

        Returns:
        --------------------------------
        Viscous flux in x and viscous flux in y 
        Units: kg/s^2
        
        """
        
        tau_filtered = self._filtered_stress()
        tau_sgs = self._sgs_stress()
        tau_net = tau_filtered + tau_sgs
        # tau has shape (2, 2, numX, numY) 
        tau_xx = tau_net[0,0] 
        tau_yy = tau_net[1,1]
        # expected large shear flux in case of couette flow
        tau_xy = tau_net[0,1]
        # calculate the divergence of the stress tensor 
        visc_flux_x = first_order_partial(tau_xx, self.mesh.dx, axis=1) + first_order_partial(tau_xy, self.mesh.dy, axis=0)
        visc_flux_y = first_order_partial(tau_xy, self.mesh.dx, axis=1) + first_order_partial(tau_yy, self.mesh.dy, axis=0)
        return (visc_flux_x, visc_flux_y)


    def calculate_intermediate_velocities(self, u:np.ndarray, v:np.ndarray, dt:float, Fconv:tuple, Fvisc:tuple) -> tuple[np.ndarray]:
        """
        intermediate velocities for the projection method of calculating pressure
        calculates intermediate velocities WITHOUT pressure term

        Parameters:
        --------------------------------
        `u` : np.ndarray
        
        `v` : np.ndarray
         
        `dt` : float

        `Fconv` : tuple
            tuple of x and y convective fluxes of u as well as the x and y fluxes of v
        `Fvisc` : tuple
            tuple of viscous x and y viscous/diffusive fluxes

        Returns
        --------------------------------


        `u* = u + (∆t/ρ)(-Fc_ux/∆x - Fc_uy/∆y + Fv_ux/∆x + Fv_uy/∆y)`
        """
        Fconv_ux, Fconv_uy, Fconv_vx, Fconv_vy = Fconv
        Fvisc_u, Fvisc_v = Fvisc
        dx, dy = self.mesh.dx, self.mesh.dy

        #                    Net velocity leaving and entering the volume
        u_star = u + dt * (Fconv_ux/dx + Fconv_uy/dy + Fvisc_u/dx)
        v_star = v + dt * (Fconv_vx/dx + Fconv_vy/dy + Fvisc_v/dy)
        return (u_star, v_star)

    def _pressure_source_term(self):
        """
        Calculates the pressure source term, which is the divergence of the fluxes.

        `b = ∇·(uu)`
        """

        # TODO boundaries
        u, v = self.mesh['u'][:,:], self.mesh['v'][:,:]
        b = divergence(u, v, self.mesh.dx, self.mesh.dy)
        return b[2:-2, 2:-2]
        
    def solvePressureCorrectionEquation(self, dt, solver:EquilibriumSolver=EquilibriumSolver.gauss_seidel, tolerance:float=1e-6) -> np.ndarray:
        """
        Solves the pressure-poisson equation using a Jacobi Iteration method

        Parameters
        --------------------------------
        dt : timestep


        solver : EquilibriumSolver
            Algebraic solver desired

        Returns
        --------------------------------
        Corrected pressure field

        ESTIMATED CHARACTERISTIC PRESSURE: 62.5 Pa = kg / (m s^2) -> (for my initial conditions)
        How I calculate the pressure: pressure laplacian = Divergence of the convection terms ->   
        ∇^2p = ∇⋅((u⋅∇)u) -> 
        this pressure calculation is used to 
        correct the velocities so that they satisfy the continuity equation -> u* - (∆t/ρ)∇p = 0

        ∂ρ/∂t + ∇⋅(ρu) = 0
        """
        # TODO Reimplement
        # matrix of the coefficients in units of 1/m^2
        A = self.mesh.A
        # matrix of the divergence of the velocity vectors
        b = self._pressure_source_term()
        b = self.mesh.rho * b / dt
        # uses the conjugare gradient method
        p = self.mesh['p'][:,:]
        p_corrected = np.zeros_like(p)
        """p_corrected = EquilibriumSolver.conjugate_gradient(A, b, p[2:-2, 2:-2], tolerance=tolerance, maxiter=5000)
        print(p_corrected)
        p_corrected = EquilibriumSolver.jacobi(A, b, p[2:-2, 2:-2], tolerance=tolerance, maxiter=5000)
        print(p_corrected)"""
        p_corrected = solver(A, b, p[2:-2, 2:-2], tolerance=tolerance, maxiter=5000)
        return p_corrected
    
    def updateVelocityField(self, u_star:np.ndarray, v_star:np.ndarray, p:np.ndarray, dt:float) -> tuple[np.ndarray]:
        """

        Parameters
        --------------------------------
        u_star : np.ndarray

        v_star : np.ndarray

        p : np.ndarray

        dt : float
        
        Returns
        --------------------------------
        """
        p = self.mesh['p'][:,:]
        grad_p_x = first_order_partial(p, self.mesh.dx, axis=1)
        grad_p_y = first_order_partial(p, self.mesh.dy, axis=0)
        # m/s     =  m/s   -   s m^3/kg     *  kg/s^2m^2
        u_updated = u_star - dt / self.mesh.rho * grad_p_x
        v_updated = v_star - dt / self.mesh.rho * grad_p_y
        return (u_updated, v_updated)