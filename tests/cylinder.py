from FluxPy.CFDSolve import *
import h5py
import os

# 2D Cylinder Geometry
center = np.array([2,2])
radius = 0.25

# A common setup would be to have the domain length 10-20 times the diameter of the cylinder in the 
# flow direction and 5-10 times the diameter in the cross-flow 
# Length and height of our domain (in meters) -> we will not be working 
# with a characteristice length scale here 
Lx, Ly = 10.0, 4.0
# Number of cells in each direction
numX, numY = 400, 400
# grid spacing (Δx, Δy not actually dx or dy)
dx = Lx/numX
dy = Ly/numY
X, Y = np.meshgrid(np.linspace(0, Lx, numX+4), np.linspace(0, Ly, numY+4))

# Density of water in kg/m^3
rho = 1000 # * u.kg/(u.m**3)
# dynamic viscosity coefficient
mu = 1.0e-3 # * u.Pa # for water -> Pa s
# kinematic viscosity coefficient
nu = 1.002e-6 # = mu / rho  # Kinematic viscosity of water in m^2/s
# bulk viscosity relation:
lam = -(2/3) * mu # lam + (2/3) * mu = 0
inlet_velocity = 2
CFL = 0.5

# Time stuff
max_dt = 0.00625
dt = 0.00625
nt = 10
t = 0
max_iters = int(nt/dt)

# Mask for only points inside the cylinder
inside = inside_cylinder(X, Y, radius, center)
# These are the default boundary conditions for vortex shedding past a cylinder
# Most simulations are extremely sensitive to changes in boundary conditions because
# they are dealing with nonlinear systems
BCS = {
    'u':{
         'left':   {'condition': Dirichlet, 'value': 1},
         'right':  {'condition': Neumann, 'value': 0, 'type': 'zerograd'}, 
         'top':    {'condition': Neumann, 'value': 0}, 
         'bottom': {'condition': Neumann, 'value': 0},
         'custom': {'boundary' : inside, 'value': 0}
         }, 
    'v':{
         'left':   {'condition': Dirichlet, 'value': 0}, 
         'right':  {'condition': Neumann, 'value': 0},
         'top':    {'condition': Neumann, 'value': 0}, 
         'bottom': {'condition': Neumann, 'value': 0},
         'custom': {'boundary' : inside, 'value': 0}
         }, 
    'p':{
         'left':   {'condition': Neumann,   'value': 0,   'type': 'zerograd'}, 
         'right':  {'condition': Dirichlet, 'value': 0,   'type': 'zerograd'}, 
         'top':    {'condition': Neumann,   'value': 0,   'type': 'zerograd'}, 
         'bottom': {'condition': Neumann,   'value': 0,   'type': 'zerograd'}
         }
    }

# Priority order in which the boundary conditions will be considered
# The simulation is also extremely sensitive to changes in this
priority_map = {
    'Neumann': 2,
    'Outflow':2,
    'NoSlip':2,
    'Dirichlet': 1,
    'Inflow':3
    }

constants =  {'rho':rho, 'mu':mu}

def main():

    # Initialize the large eddy simualtion class
    solver = LES((Lx, Ly), (numX, numY), BCS, nghosts=2, constants=constants)
    shape = (int(nt/dt), solver.mesh['u'][:,:].shape[0], solver.mesh['u'][:,:].shape[1])
    iter = 0
    # This is for collecting the data in case you want to process it later
    with h5py.File(f"{os.path.dirname(__file__)}/run1.h5", 'w') as f:
        dset_u = f.create_dataset("u", shape, dtype=np.float64)
        dset_v = f.create_dataset("v", shape, dtype=np.float64)
        dset_p = f.create_dataset("p", shape, dtype=np.float64)
        while iter < max_iters:
            dt = min(max_dt, CFL * min(solver.mesh.dx, solver.mesh.dy) / (np.max(np.sqrt(solver.mesh['u'][:,:]**2 + solver.mesh['v'][:,:]**2))) + 1e-9)
            solver.mesh.apply_all_boundaries()

            # Filter the mesh
            solver.mesh['u'][:,:] = Filter.gaussian(solver.mesh['u'][:,:], sigma=solver.mesh.dx)
            solver.mesh['v'][:,:] = Filter.gaussian(solver.mesh['v'][:,:], sigma=solver.mesh.dx)

            # Calculate the Fluxes
            conv_flux_x_u, conv_flux_y_u = solver._convective_flux('u')
            conv_flux_x_v, conv_flux_y_v = solver._convective_flux('v')
            Fvu, Fvv = solver._viscous_flux()
            
            # Get intermediate velocities
            u_star, v_star = solver.calculate_intermediate_velocities(solver.mesh['u'][:,:], solver.mesh['v'][:,:], dt, (conv_flux_x_u, conv_flux_y_u, conv_flux_x_v, conv_flux_y_v), (Fvu, Fvv))
            solver.mesh['u'][:,:], solver.mesh['v'][:,:] = u_star, v_star 
            solver.mesh.apply_all_boundaries()        

            # Solve for the pressure correction term 
            p_new = solver.solvePressureCorrectionEquation(dt)
            solver.mesh['p'][2:-2,2:-2] = p_new
            solver.mesh.apply_all_boundaries()    

            # Update the velocity fields
            u_new, v_new = solver.updateVelocityField(solver.mesh['u'][:,:], solver.mesh['v'][:,:],p_new,dt)
            solver.mesh['u'][:,:], solver.mesh['v'][:,:] = u_new, v_new

            # Log the data
            dset_u[iter, :, :] = solver.mesh['u'][:,:]
            dset_v[iter, :, :] = solver.mesh['v'][:,:]
            dset_p[iter, :, :] = solver.mesh['p'][:,:]
            if iter % 5 == 0:
                path = f"/home/jacob.schaefer/SogGitlab/LES/figs/u_velocity_figs/fig{iter}.png"

                # vorticity
                omega = np.abs(curl(solver.mesh['u'][2:-2,2:-2],solver.mesh['v'][2:-2,2:-2],dx,dy))

                # velocity magnitude
                mag = np.sqrt(solver.mesh['u'][2:-2,2:-2]**2 + solver.mesh['v'][2:-2,2:-2]**2)

                field = solver.mesh['u'][2:-2,2:-2]
                plot_field(solver.mesh.X[2:-2,2:-2], solver.mesh.Y[2:-2,2:-2], field, solver.mesh.size[0], solver.mesh.size[1], save=True, path = path, show=True)
            t+=dt
            iter +=1


if __name__ == '__main__':
    main()

