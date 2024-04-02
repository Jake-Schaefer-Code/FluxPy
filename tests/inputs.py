from FluxPy.CFDSolve import *
from FluxPy.boundaries import *
from FluxPy.utils.plotting import *
import numpy as np

# 2D Cylinder Geometry
center = np.array([1.5,2])
radius = 0.25

# A common setup would be to have the domain length 10-20 times the diameter of the cylinder in the 
# flow direction and 5-10 times the diameter in the cross-flow 
# Length and height of our domain (in meters) -> we will not be working 
# with a characteristice length scale here 
Lx, Ly = 12.0, 4.0
# Number of cells in each direction
numX, numY = 200, 200
# grid spacing (Δx, Δy not actually dx or dy)
dx = Lx/numX
dy = Ly/numY
X, Y = np.meshgrid(np.linspace(0, Lx, numX+4), np.linspace(0, Ly, numY+4))

# Density of water in kg/m^3
rho = 1000 # * u.kg/(u.m**3)
# dynamic viscosity coefficient
mu = 1.0e-3 # * u.Pa # for water -> Pa s
inlet_velocity = 2
CFL = 0.5

# Time stuff
max_dt = 0.00625
dt = 0.00625
nt = 10000
max_iters = int(nt/dt)

# THESE ARE THE BCS FOR VORTEX SHEDDING PAST A CYLINDER
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
        'right':  {'condition': Neumann, 'value': 0, 'type': 'zerograd'},
        'top':    {'condition': Neumann, 'value': 0}, 
        'bottom': {'condition': Neumann, 'value': 0},
        'custom': {'boundary' : inside, 'value': 0}
        }, 
    'p':{
        'left':   {'condition': Neumann,   'value': 0,   'type': 'zerograd'}, 
        'right':  {'condition': Dirichlet, 'value': 0}, 
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