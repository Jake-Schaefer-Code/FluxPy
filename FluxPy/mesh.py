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

        `prim` : str

        `BCS` : dict{str : BoundaryCondtition} 
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
        Parametes
        --------------------------------
        `BC1` : BoundaryCondition

        `BC2` : BoundaryCondition


        """
        intersection = Slices.get_slice_intersect(BC1.indices, BC2.indices, self.mesh.ncells)
        if intersection[0] is None or intersection[1] is None:
            return
        elif _priority_map[str(BC1)] > _priority_map[str(BC2)]:
            BC1.indices = Slices.adjust_indices(BC1.indices, intersection, self.mesh.ncells, (BC1.boundary, BC2.boundary))
        elif _priority_map[str(BC1)] < _priority_map[str(BC2)]:
            BC2.indices = Slices.adjust_indices(BC2.indices, intersection, self.mesh.ncells, (BC2.boundary, BC1.boundary))
        # TODO figure this one out
        else:
            BC2.indices = Slices.adjust_indices(BC2.indices, intersection, self.mesh.ncells, (BC2.boundary, BC1.boundary))

    def _check_adjacency(self, edge1, edge2):
        return

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

        `value` : float

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


    def apply_boundary_faces(self, phi_e, phi_w, phi_n, phi_s):
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
    Overarching Mesh Class
    """
    def __init__(self, size:tuple, ncells:tuple, BCS:dict={}, nghosts:int=1, constants:dict={}) -> None:
        """
        Parameters
        --------------------------------
        `size` : tuple

        `ncells` : tuple

        `BCS` : dict = {}

        `nghosts` : int = 1

        `constants` : dict = {}
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
        initializes kwargs constants
        """
        self.rho = constants.get('rho', 1.0)
        self.mu = constants.get('mu', 1.0)

    def _initialize_boundaries(self, BCS:dict[str, dict[str, BoundaryCondition]]) -> None:
        """
        Initializes the boundaries for a rectangular domain. Default boundary condition is Dirichlet.

        Parameters
        --------------------------------
            BCS : dict{str: {str: BoundaryCondition}}
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
        """
        self.BCS[prim] = boundary
    
    def apply_all_boundaries(self) -> None:
        for prim in ['u', 'v', 'p']:
            self[prim].apply_boundaries()

    def __getitem__(self, prim:str) -> Field:
        """
        
        """
        if isinstance(prim, str):
            return getattr(self, prim)
    
    def __setitem__(self, prim, value, stencil=None):
        """
        Parameters
        --------------------------------
        prim : str

        value : 

        stencil : 
        """
        field = getattr(self,prim)
        if stencil is None:
            field[:,:] = value
        else:
            field[stencil] = value