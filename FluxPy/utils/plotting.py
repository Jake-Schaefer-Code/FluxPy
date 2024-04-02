import numpy as np
# import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .PDE import curl

LOGO = "\n\
                ________           ____       \n\
               / ____/ /_  ___  __/ __ |__  __\n\
              / /_  / / / / / |/_/ /_/ / / / /\n\
             / __/ / / /_/ />  </ ____/ /_/ / \n\
            /_/   /_/\__,_/_/|_/_/    \__, /  \n\
                                     /____/   \n"

def plot_sim(X:np.ndarray, Y:np.ndarray, u:np.ndarray, v:np.ndarray, p:np.ndarray, Lx, Ly, dx, dy, iter, 
             stream:bool=True, quiv:bool=False, grid_lines=False, titles:bool=False, savefig:bool=False, **kwargs):
    """
    Plots simulation results
    
    Parameters
    --------------------------------
    X : np.ndarray 
    
    Y : np.ndarray 
    
    u : np.ndarray 
    
    v : np.ndarray 
    
    p : np.ndarray 
    
    Lx 
    
    Ly 
    
    dx 
    
    dy 
    
    iter 
    
    stream : bool=True 
    
    quiv : bool=False 
    
    grid_lines=False 
    
    titles : bool=False 
    
    savefig : bool=False 
    
    **kwargs

    Returns
    --------------------------------
    """
    v_mag = np.sqrt(u**2 + v**2)
    omega = curl(u,v,dx,dy)
    fig1= plt.figure(1, figsize=(Lx, Ly))
    sp1 = fig1.add_subplot(111)
    sp1.set_aspect('equal')
    sp1.set_xlim(0, Lx)
    sp1.set_ylim(0, Ly)
    levels = np.linspace(np.min(omega), np.max(omega), 100)
    try:
        contour1 = sp1.contourf(X, Y, omega, levels=levels, cmap='jet')
    except:
        contour1 = sp1.contourf(X, Y, omega, levels=100, cmap='jet')
    if quiv:
        stride = 16
        sp1.quiver(X[::stride, ::stride], Y[::stride, ::stride], (u/np.linalg.norm(u))[::stride, ::stride], (v/np.linalg.norm(v))[::stride, ::stride])
    if stream:
        sp1.streamplot(X, Y, u, v, density = 3, color="red")
    if grid_lines:
        x = np.linspace(0, Lx, u.shape[1])
        y = np.linspace(0, Ly, u.shape[0])
        for i in range(u.shape[1]):
            sp1.plot([x[i], x[i]], [y[0], y[-1]], color='black', linewidth=0.5)
        for j in range(u.shape[0]):
            sp1.plot([x[0], x[-1]], [y[j], y[j]], color='black', linewidth=0.5)
    if titles:
        cbar1 = fig1.colorbar(contour1)
    if savefig:
        plt.savefig(f'/home/jacob.schaefer/SogGitlab/vorticityfigs/vorticity_plot{iter}')
    return fig1


def plot_field(X:np.ndarray, Y:np.ndarray, phi:np.ndarray, Lx, Ly, save=False, path=None, show=True):
    """
    Plots input field

    Parameters
    --------------------------------

    Returns
    --------------------------------
    """
    fig1= plt.figure(1, figsize=(2*Lx, 2*Ly))
    sp1 = fig1.add_subplot(111)
    sp1.set_aspect('equal')
    sp1.set_xlim(0, Lx+0.1)
    sp1.set_ylim(0, Ly+0.1)
    levels = np.linspace(np.min(phi), np.max(phi), 100)
    contour1 = sp1.contourf(X, Y, phi, levels=levels, cmap='bone')
    cbar1 = fig1.colorbar(contour1)
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    return fig1