from boundaries import *
from utils import make_sparse_A

_required_edges = ['left', 'right', 'top', 'bottom']
_default_boundary = {'condition': Dirichlet, 'value': 0.0}
_default_boudary_dict =  {'left': _default_boundary, 
                        'right': _default_boundary,
                        'top': _default_boundary,
                        'bottom': _default_boundary}
_default_prim_boundary_dict = {'u': _default_boudary_dict, 
                               'v': _default_boudary_dict, 
                               'p': _default_boudary_dict} 
_priority_map = {
    'Neumann': 2,
    'Outflow':2,
    'NoSlip':2,
    'Dirichlet': 1,
    'Inflow':3
}



class Field(FieldInterface):
    """
    Field class for the field of a primitive scalar
    """
    def __init__(self, mesh:MeshInterface, prim:str, BCS:dict[str, BoundaryCondition]) -> None:
        """
        Parameters
        --------------------------------
        `mesh` : Mesh(MeshInterface)
            Mesh object
        `prim` : str
            Primitive key / field type
        `BCS` : dict{str : BoundaryCondtition} 
            Boundary condition dictionary
        """
        self.mesh = mesh
        self.prim = prim
        self.BCS:dict[str, BoundaryCondition] = BCS
        self.field = np.zeros(mesh.ncells)
        self.edges = _required_edges
        if 'custom' in self.BCS:
            self.custom_bcs = [self.BCS['custom']]
            del self.BCS['custom']
        else:
            self.custom_bcs = []
        
        self._setup_boundaries()
        for edge in _required_edges:
            self.BCS[edge]._get_edge_layer_indices()


    def _setup_boundaries(self):
        edges = list(self.BCS.keys())
        for i, edge1 in enumerate(edges):
            bc1 = self.BCS[edge1]
            for edge2 in edges[i+1:]:
                bc2 = self.BCS[edge2]
                self._resolve_overlap(bc1, bc2)

    def _resolve_overlap(self, BC1:BoundaryCondition, BC2:BoundaryCondition):
        """
        Crops boundary conditions based on priority if they overlap

        Parameters
        --------------------------------
        `BC1` : BoundaryCondition
            Boundary condition 1
        `BC2` : BoundaryCondition
            Boundary condition 2
        """
        intersection = Slices.get_slice_intersect(BC1.indices, BC2.indices, self.mesh.ncells)
        if intersection[0] is None or intersection[1] is None:
            return
        elif self.mesh.priority_map[str(BC1)] > self.mesh.priority_map[str(BC2)]:
            BC1.indices = Slices.adjust_indices(BC1.indices, intersection, self.mesh.ncells, (BC1.boundary, BC2.boundary))
        elif self.mesh.priority_map[str(BC1)] < self.mesh.priority_map[str(BC2)]:
            BC2.indices = Slices.adjust_indices(BC2.indices, intersection, self.mesh.ncells, (BC2.boundary, BC1.boundary))
        # TODO figure this one out
        else:
            BC2.indices = Slices.adjust_indices(BC2.indices, intersection, self.mesh.ncells, (BC2.boundary, BC1.boundary))

    def __getitem__(self, stencil) -> np.ndarray:
        """
        Given a stencil, this returns part of the field -> Field[stencil]
 
        Parameters
        --------------------------------
        `stencil` : str | np.ndarray | tuple | int
            Stencil is intersected with the array, giving a subset

        Returns
        --------------------------------
        A subset of the array
        """
        nghosts = self.mesh.nghosts
        if isinstance(stencil, np.ndarray) or isinstance(stencil, int) or isinstance(stencil, tuple):
            subset = self.field[stencil]
        elif stencil == 'left':
            subset = self.field[:,:nghosts]
        elif stencil == 'right':
            subset = self.field[:,-nghosts:]
        elif stencil == 'top':
            subset = self.field[-nghosts:,:]
        elif stencil == 'bottom':
            subset = self.field[:nghosts,:]
        return subset
    
    
    def __setitem__(self, stencil:np.ndarray, value:float) -> None:
        """
        Parameters
        --------------------------------
        `stencil` : np.ndarray | int | tuple
            Mask for where to apply value
        `value` : float | np.ndarray
            Value to set item to

        Returns
        --------------------------------
        None
        """
        self.field[stencil] = value

    def __str__(self):
        out = f"{self.prim}:\n"
        for edge, bc in self.BCS.items():
            out += f"{edge}: {bc}\n"
        out += str(self.field)
        return out
    
    def apply_boundaries(self):
        for edge, bc in self.BCS.items():
            # if edge != 'left' and edge != 'right':
            bc.apply(self, dx=self.mesh.dx)
        if self.custom_bcs:
            for bc, val in self.custom_bcs:
                self.field[bc] = val


    def apply_boundary_faces(self, phi_e:np.ndarray, phi_w:np.ndarray, phi_n:np.ndarray, phi_s:np.ndarray):
        """
        Applies boundary conditions to each directional face

        Parameters
        --------------------------------
        `phi_e` : np.ndarray
            East face
        `phi_w` : np.ndarray
            West face
        `phi_n` : np.ndarray
            North face
        `phi_s` : np.ndarray
            South face

        Returns
        --------------------------------
        Faces with applied boundary conditions
        """
        nghosts = self.mesh.nghosts
        dx,dy=self.mesh.dx,self.mesh.dy
        for edge, bc in self.BCS.items():
            if edge == 'left':
                phi_e = bc.apply(phi_e, value=phi_e[:, nghosts], dx=dx)
                phi_w = bc.apply(phi_w, value=phi_w[:, nghosts], dx=dx)
            elif edge == 'right':                
                phi_e = bc.apply(phi_e, value=phi_e[:, -nghosts-1], dx=dx)
                phi_w = bc.apply(phi_w, value=phi_w[:, -nghosts-1], dx=dx)
            elif edge == 'top':
                phi_n = bc.apply(phi_n, value=phi_n[-nghosts-1, :], dx=dy)
                phi_s = bc.apply(phi_s, value=phi_s[-nghosts-1, :], dx=dy)
            elif edge == 'bottom':
                phi_n = bc.apply(phi_n, value=phi_n[nghosts, :], dx=dy)
                phi_s = bc.apply(phi_s, value=phi_s[nghosts, :], dx=dy)
        return (phi_e, phi_w, phi_n, phi_s)


class Mesh(MeshInterface):
    """
    Mesh Class
    """
    def __init__(self, size:tuple, ncells:tuple, BCS:dict={}, priority_map:dict=_priority_map, nghosts:int=1, constants:dict={}) -> None:
        """
        Parameters
        --------------------------------
        `size` : tuple
            Size of the mesh in form (x,y)
        `ncells` : tuple
            Number of cells in the x and y directions
        `BCS` : dict
            Boundary condition dictionary. Defaults to {}
        `priority_map` : dict
            Priorities which to give boundary conditions during applicaiton.
        `nghosts` : int = 1
            Number of ghost cells to add to each side
        `constants` : dict
            Constants dictionary. Defaults to {}
        """
        self.size = size
        self.ncells = ncells
        self.dx = size[0] / ncells[0]
        self.dy = size[1] / ncells[1]
        self.ncells = (ncells[0]+2*nghosts, ncells[1]+2*nghosts)
        self.X, self.Y = np.meshgrid(np.linspace(0, self.size[0], self.ncells[0]), 
                                     np.linspace(0, self.size[1], self.ncells[1]))
        # Accounts for the boundary layer that is not technically a ghost cell
        self.nghosts=nghosts+1
        self.BCS:dict[str, dict[str, dict[str, BoundaryCondition]]] 
        self.priority_map = priority_map
        self._initialize_constants(constants)
        self._initialize_boundaries(BCS)
        
        self.u = Field(self, 'u', self.BCS['u'])
        self.v = Field(self, 'v', self.BCS['v'])
        self.p = Field(self, 'p', self.BCS['p'])
        self.apply_all_boundaries()
        self.A = make_sparse_A(self.ncells[0]-4, self.ncells[1]-4, self.dx, self.dy)


    # TODO ----------------
        # Instead of setting attr, rho and mu explicitely
    def _initialize_constants(self, constants:dict) -> None:
        """
        Initializes constants

        Parameters
        --------------------------------
        `constants` : dict
        """
        self.rho = constants.get('rho', 1.0)
        self.mu = constants.get('mu', 1.0)

    def _initialize_boundaries(self, BCS:dict[str, dict[str, BoundaryCondition]]) -> None:
        """
        Initializes the boundaries for a rectangular domain. Default boundary condition is Dirichlet.

        Parameters
        --------------------------------
        `BCS` : dict{str: {str: BoundaryCondition}}
        """
        
        self.BCS = {'u':{}, 'v':{}, 'p':{}}
        if not BCS: BCS = _default_prim_boundary_dict
        for prim, edges in BCS.items():
            for edge in _required_edges:
                boundary = edges.get(edge, _default_boundary)
                bc_class = boundary.get('condition', Dirichlet)
                bc_value = boundary.get('value', 0.0)
                bc_type = boundary.get('type', None)
                bc = bc_class(edge, bc_value, bc_type, nghosts=self.nghosts)
                self.BCS[prim][edge] = bc
            if 'custom' in edges:
                self.BCS[prim]['custom'] = (edges['custom']['boundary'], edges['custom']['value'])


# BCS[prim][edge] = BCS[prim].get(edge, (Dirichlet, 0))
    def update_boundary(self, prim:str, boundary:dict[str, BoundaryCondition]) -> None:
        """
        Updates the given boundary condition for a primitive

        Parameters
        --------------------------------
        `prim` : str
            The primitive field to update
        """
        self.BCS[prim] = boundary
    
    def apply_all_boundaries(self) -> None:
        for prim in ['u', 'v', 'p']:
            self[prim].apply_boundaries()

    def __getitem__(self, prim:str) -> Field:
        """
        Parameters
        --------------------------------
        `prim` : str
            Primitive key for field
        """
        if isinstance(prim, str):
            return getattr(self, prim)
    
    def __setitem__(self, prim:str, value:float|np.ndarray, stencil:np.ndarray=None):
        """
        Parameters
        --------------------------------
        `prim` : str
            Primitive key for field
        `value` : float | np.ndarray
            The value(s) to apply to the primitive field
        `stencil` : np.ndarray
            Mask on which to apply value
        """
        field = getattr(self,prim)
        if stencil is None:
            field[:,:] = value
        else:
            field[stencil] = value