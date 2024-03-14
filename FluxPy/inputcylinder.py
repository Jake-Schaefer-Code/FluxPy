from CFDSolve import *
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
numX, numY = 800, 800
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
inlet_velocity = 4
CFL = 0.5

# Time stuff
max_dt = 0.00625
dt = 0.00625
nt = 10
max_iters = int(nt/dt)




def save_to_file(file_index:int, data:dict):
    with h5py.File(f'/home/jacob.schaefer/SogGitlab/FluxPy/FluxPy/data2/data_{file_index}.h5', 'w') as f:
        for dataset_name, dataset_values in data.items():
            f.create_dataset(dataset_name, data=dataset_values, dtype=np.float64, chunks=True)


def main():
    # 2D Cylinder Geometry
    center = np.array([2,2])
    radius = 0.25

    # A common setup would be to have the domain length 10-20 times the diameter of the cylinder in the 
    # flow direction and 5-10 times the diameter in the cross-flow 
    # Length and height of our domain (in meters) -> we will not be working 
    # with a characteristice length scale here 
    Lx, Ly = 10.0, 4.0
    # Number of cells in each direction
    numX, numY = 800, 800
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
    inlet_velocity =2
    CFL = 0.5

    # Time stuff
    max_dt = 0.00625
    dt = 0.00625
    nt = 1000
    max_iters = int(nt/dt)

    # Mask for only points inside the cylinder
    inside = inside_cylinder(X, Y, radius, center)
    # These are the default boundary conditions for vortex shedding past a cylinder
    # Most simulations are extremely sensitive to changes in boundary conditions because
    # they are dealing with nonlinear systems
    BCS = {
        'u':{
            'left':   {'condition': Dirichlet, 'value': inlet_velocity},
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

    # Initialize the large eddy simualtion class
    solver = LES((Lx, Ly), (numX, numY), BCS, priority_map, nghosts=2, constants=constants)
    shape = (int(nt/dt), solver.mesh['u'][:,:].shape[0], solver.mesh['u'][:,:].shape[1])
    iter = 0
    t = 0
    path = "/home/jacob.schaefer/SogGitlab/FluxPy/FluxPy"
    # This is for collecting the data in case you want to process it later

    data_dict = {'u':[], 'v':[], 'p':[]}
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
        # There are currently 3 different solvers that I have implemented, but I have found 
        # that gauss_seidel is the most accurate
        p_new = solver.solvePressureCorrectionEquation(dt, solver=EquilibriumSolver.gauss_seidel)
        solver.mesh['p'][2:-2,2:-2] = p_new
        solver.mesh.apply_all_boundaries()    

        # Update the velocity fields
        u_new, v_new = solver.updateVelocityField(solver.mesh['u'][:,:], solver.mesh['v'][:,:],p_new,dt)
        solver.mesh['u'][:,:], solver.mesh['v'][:,:] = u_new, v_new

        # Log the data
        data_dict['u'].append(solver.mesh['u'][:,:])
        data_dict['u'].append(solver.mesh['v'][:,:])
        data_dict['u'].append(solver.mesh['p'][:,:])
        if iter % 50 == 0:
            save_to_file(iter//50, data_dict)
            data_dict = {'u':[], 'v':[], 'p':[]}

        if iter % 5 == 0:
            path_u = f"{path}/figs/u_velocity_figs/fig{iter}.png"
            plot_field(solver.mesh.X[2:-2,2:-2], solver.mesh.Y[2:-2,2:-2], solver.mesh['u'][2:-2,2:-2], solver.mesh.size[0], solver.mesh.size[1], save=True, path=path_u, show=False)
            path_p = f"{path}/figs/pressure_figs/fig{iter}.png"

            # vorticity
            plot_field(solver.mesh.X[2:-2,2:-2], solver.mesh.Y[2:-2,2:-2], solver.mesh['p'][2:-2,2:-2], solver.mesh.size[0], solver.mesh.size[1], save=True, path=path_p, show=False)
            path_omega = f"{path}/figs/vortex_figs/fig{iter}.png"
            omega = np.abs(curl(solver.mesh['u'][2:-2,2:-2],solver.mesh['v'][2:-2,2:-2],dx,dy))
            plot_field(solver.mesh.X[2:-2,2:-2], solver.mesh.Y[2:-2,2:-2], omega, solver.mesh.size[0], solver.mesh.size[1], save=True, path=path_omega, show=True)
            
            # velocity magnitude    
            path_mag = f"{path}/figs/mag_velocity_figs/fig{iter}.png"
            mag = np.sqrt(solver.mesh['u'][2:-2,2:-2]**2 + solver.mesh['v'][2:-2,2:-2]**2)
            plot_field(solver.mesh.X[2:-2,2:-2], solver.mesh.Y[2:-2,2:-2], mag, solver.mesh.size[0], solver.mesh.size[1], save=True, path=path_mag, show=True)
            
        t+=dt
        iter +=1


if __name__ == '__main__':
    main()

