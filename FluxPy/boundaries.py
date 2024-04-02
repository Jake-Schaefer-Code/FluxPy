import numpy as np
from .interface import *
from .utils.utils import *
np.set_printoptions(precision=3, linewidth=150)



class BoundaryCondition(BoundaryInterface):
    """
    Parent class of all boundary condition subclasses
    """
    def __init__(self, boundary, value, flow_type:str=None, nghosts:int=1):
        self.boundary = boundary
        self.axis = self._get_axis(boundary)
        
        # TODO compute tangents
        # self.tangent_dir = 0
        self.dir = self._get_face_normal(boundary)

        self.value = value
        self.flow_type = flow_type
        self.nghosts = nghosts
        self.indices = self._get_indices(nghosts)
        self.edge_layer = 0
    
    def _get_axis(self, boundary):
        """
        
        """
        if boundary in ['left', 'right']:
            axis = 1
        elif boundary in ['top', 'bottom']:
            axis = 0
        else:
            axis = None
        return axis

    def _get_face_normal(self, face:np.ndarray|str) -> float:
        """
        
        """
        if face in ['left', 'bottom']:
            dir = 1
        elif face in ['right', 'top']:
            dir = -1
        else:
            dir = None
        return dir
            
    def _get_indices(self, nghosts:int=1):
        """
        Parameters:
        --------------------------------
        `boundary` : str | int | np.ndarray | tuple
            The boundary to get the indices of
        `nghosts` : int=1
            Number of ghost cells to consider
            
        Returns
        --------------------------------
        Indices of the boundary
        """
        
        """
        boundary = self.boundary
        if boundary == 'left':
            indices = (slice(0, None), slice(0, nghosts))
        elif boundary == 'right':
            indices = (slice(0, None), slice(-nghosts, None))
        elif boundary == 'top':
            indices = (slice(-nghosts, None), slice(0, None))
        elif boundary == 'bottom':
            indices = (slice(0, nghosts), slice(0, None))"""
        if self.dir is not None: 
            boundary_slice = slice(0, nghosts) if self.dir == 1 else slice(-nghosts, None)
            indices = (boundary_slice, slice(0, None)) if self.axis == 0 else (slice(0, None), boundary_slice)
        elif self.__class__ is not Neumann and isinstance(self.boundary, (np.ndarray, tuple, int)):
            indices = self.boundary
        else:
            raise NotImplementedError(f"{self.boundary} is not a known boundary for type {self.__class__}")

        return indices

    def _get_gradient(self, field, dx):
        if self.dir == 1:
            idx = self.nghosts
        elif self.dir == -1:
            idx = self.nghosts + 1

        if self.axis == 0:
            grad = (field[self.dir * idx, self.indices[1]] - field[self.dir * (idx-1), self.indices[1]]) / dx
        elif self.axis == 1:
            grad = (field[self.indices[0], self.dir * idx] - field[self.indices[0], self.dir * (idx-1)]) / dx
        else:
            raise NotImplementedError(f"Gradient for boundary {self.boundary} is not implemented for type {self.__class__}")

        """if self.boundary == 'left':
            grad = ((field[:, self.nghosts] - field[:, self.nghosts-1]) / dx)[self.indices[0]]
        elif self.boundary == 'right':
            grad = ((field[:, -(self.nghosts+1)] - field[:, -(self.nghosts)]) / dx)[self.indices[0]]
        elif self.boundary == 'top':
            grad = ((field[-(self.nghosts + 1), :] - field[-(self.nghosts + 1 - 1), :]) / dx)[self.indices[1]]
        elif self.boundary == 'bottom':
            grad = ((field[self.nghosts, :] - field[self.nghosts-1, :]) / dx)[self.indices[1]]
        else:
            raise NotImplementedError(f"{self.boundary} is not a known boundary for type {self.__class__}")"""
        
        return grad
    
    def _apply_constant_grad(self, field, dx):
        nghosts = self.nghosts
        # grad_value = self._get_gradient(field, dx) if self.flow_type == 'constant' else 0
        grad_value = self._get_gradient(field, dx)
        # NEW
        idx = nghosts - 1        
        if self.boundary == 'left':
            for i in range(nghosts): 
                field[self.indices[0], i] -= grad_value * dx * (idx - i)

        elif self.boundary == 'right':
            for i in range(nghosts):
                field[self.indices[0], -(i+1)] -= grad_value * dx * (idx - i)

        elif self.boundary == 'top':            
            for i in range(nghosts):
                field[-(i+1), self.indices[1]] -= grad_value * dx * (idx - i)

        elif self.boundary == 'bottom':
            for i in range(nghosts): 
                field[i, self.indices[1]] -= grad_value * dx * (idx - i)
        else:
            raise NotImplementedError(f"Gradient for boundary {self.boundary} could not be applied for type {self.__class__}")

    
        """if self.boundary == 'left':
            for i in range(nghosts): 
                field[self.indices[0], i] -= grad_value * dx * (nghosts - 1 - i)
    
        elif self.boundary == 'right':
            for i in range(1, nghosts+1):
                field[self.indices[0], -i] -= grad_value * dx * (nghosts - i)

        elif self.boundary == 'top':
            for i in range(1, nghosts+1):
                field[-i, self.indices[1]] -= grad_value * dx * (nghosts - i)

        elif self.boundary == 'bottom':
            for i in range(nghosts): 
                field[i, self.indices[1]] -= grad_value * dx * (nghosts - 1 - i)"""

        return field

    def _get_edge_layer_indices(self):
        # NEW
        if self.dir == 1:
            idx = self.nghosts
        elif self.dir == -1:
            idx = - (self.nghosts + 1)

        if self.axis == 0:
            self.edge_layer = (idx, self.indices[1])
        elif self.axis == 1:
            self.edge_layer = (self.indices[0], idx)
        else:
            raise NotImplementedError(f"Edge layer for boundary {self.boundary} is not implemented for type {self.__class__}")
        
        """if self.boundary == 'left':
            self.edge_layer = (self.indices[0], self.nghosts)
        elif self.boundary == 'right':
            self.edge_layer = (self.indices[0], -(self.nghosts+1))
        elif self.boundary == 'top':
            self.edge_layer = (-(self.nghosts+1), self.indices[1])
        elif self.boundary == 'bottom':
            self.edge_layer = (self.nghosts, self.indices[1])
        else:
            raise NotImplementedError(f"Edge layer for boundary {self.boundary} is not implemented for type {self.__class__}")"""
        return None

    def apply(self, field:FieldInterface | np.ndarray, value:np.ndarray = None, dx:float=1.0):
        """
        Applies the boundary condition to the field

        Parameters:
        --------------------------------
        `field` : Field(FieldInterface)
            Scalar primitive field
        `boundary` : str | int | np.ndarray | tuple
            The boundary to apply the condition to 
        `value` : np.ndarray | float
            Value to apply to the subset
        `nghosts` : int=1
            Number of ghost cells to consider

        Returns
        --------------------------------
        Field with applied boundary conditions
        """

        if value is None: value = self.value
        if value is None: return field

        if not isinstance(value, np.ndarray):
            field[self.indices] = value
            return field

        if self.axis == 0:
            field[self.indices] = value[self.indices[1]]
        elif self.axis == 1:
            field[self.indices] = value[self.indices[0], np.newaxis]

        
        """if self.flow_type == 'zerograd':
            # TODO CHECK THIS
            if self.axis == 0:
                # self.edge_layer = (idx, self.indices[1])
                # 
                field[self.indices] = field[self.edge_layer]
            elif self.axis == 1:
                values = field[self.indices[0], self.edge_layer[1]]
                field[self.indices] = values[:, np.newaxis]
            return field

        if not isinstance(value, np.ndarray):
            field[self.indices] = value
            return field

        if self.axis == 0:
            field[self.indices] = value[self.indices[1]]
        elif self.axis == 1:
            field[self.indices] = value[self.indices[0], np.newaxis]
        
        if self.flow_type == 'constant':
            field = self._apply_constant_grad(field, dx)"""

        """if self.boundary in ['top', 'bottom']:
            field[self.indices] = value[self.indices[1]] if isinstance(value, np.ndarray) else value
        elif self.boundary in ['left', 'right']:
            field[self.indices] = value[self.indices[0], np.newaxis] if isinstance(value, np.ndarray) else value"""

        return field
    

class Dirichlet(BoundaryCondition):
    """
    Specified velocity
    """
    def __init__(self, boundary, value, flow_type, nghosts):
        super().__init__(boundary, value, flow_type, nghosts)

    def apply(self, field:FieldInterface, value:float=None, dx:float=1.0):
        """
        Parameters
        --------------------------------
        `field` : Field(FieldInterface)
            Scalar primitive field
        `value` : np.ndarray | float
            Value to apply to the subset

        Returns
        --------------------------------
        """
        return super().apply(field, value)
    
    def __str__(self):
        return "Dirichlet"

class Neumann(BoundaryCondition):
    """
    Flux specified, Zero-gradient or constant-gradient condition
    """
    def __init__(self, boundary, value, flow_type='zerograd', nghosts:int=1):
        """
        Creates an instance of the Outflow class with a designated flow gradient type

        Parameters
        --------------------------------
        `flowtype` : str
            Type of flow. Currently can be `zerograd` or `constant`
        """
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)

    def apply(self, field:FieldInterface, value:float=None, dx:float=1.0):
        """
        Parameters
        --------------------------------
        `field` : FieldInterface
            The field to which the boundary condition will be applied
        `boundary` : str
            The boundary on which the condition will be applied
        `value` : float = 0.0   
            Value at the first boundary layer
        `nghosts` : int = 1 
            Number of ghost cells (includes non-ghost boundary layer)
        `dx` : float = 1.0
            Grid spacing
        
        Returns
        --------------------------------
        """
        if value is None: value = self.value
        if value is None: return field
        if self.flow_type == 'zerograd':
            # TODO CHECK THIS
            if self.boundary in ['top', 'bottom']:
                field[self.indices] = field[self.edge_layer]
            elif self.boundary in ['left', 'right']:
                values = field[self.edge_layer[0], self.edge_layer[1]]
                field[self.indices] = values[:, np.newaxis]
            return field

        if self.boundary in ['top', 'bottom']:
            field[self.indices] = value[self.indices[1]] if isinstance(value, np.ndarray) else value
        elif self.boundary in ['left', 'right']:
            field[self.indices] = value[self.indices[0], np.newaxis] if isinstance(value, np.ndarray) else value

        field = self._apply_constant_grad(field, dx)
        return field
    
    def __str__(self):
        return "Neumann"

class Inflow(BoundaryCondition):
    """
    Inflow boundary condition
    """
    def __init__(self, boundary, value, flow_type:str='zerograd', nghosts:int=1):
        """
        Creates an instance of the Inflow class with a designated flow gradient type

        Parameters
        --------------------------------
        `flowtype` : str
        """
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)

    def apply(self, field:FieldInterface, value:float=None, dx:float=1.0):
        """
        Parameters
        --------------------------------
        `field` : FieldInterface
        
        `value` : float = 0.0
        

        Returns
        --------------------------------
        """
        return super().apply(field, value)
    
    def __str__(self):
        return "Inflow"

class Outflow(Neumann):
    """
    Outflow boundary condition
    """
    def __init__(self, boundary, value:float|np.ndarray, flow_type:str='zerograd', nghosts:int=1):
        """
        Creates an instance of the Outflow class with a designated flow gradient type

        Parameters
        --------------------------------
        `boundary` : 

        `value` : float | np.ndarray

        `flow_type` : str
            Type of flow. Currently can be `zerograd` or `constant`
        `nghosts` : int
            Default is 1
        """
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)
 

    def __str__(self):
        return "Outflow"

    def apply(self, field:FieldInterface, value:float=None, dx:float=1.0):
        """
        Parameters
        --------------------------------
        `field` : FieldInterface
        
        `value` : float | np.ndarray
            Default is None

        Returns
        --------------------------------
        """
        return super().apply(field, value)

class NoSlip(Neumann):
    """
    No-Slip boundary condition (equal to wall velocity)
    """
    def __init__(self, boundary, value:float|np.ndarray, flow_type:str='zerograd', nghosts:int=1):
        """
        Creates an instance of the NoSlip class with a designated flow gradient type

        Parameters
        --------------------------------
        `boundary` : 

        `value` : float | np.ndarray

        `flow_type` : str
            Type of flow. Currently can be `zerograd` or `constant`
        `nghosts` : int
            Default is 1
        """
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)



    def apply(self, field:FieldInterface, value:float=None, dx:float=1.0):
        """
        Parameters
        --------------------------------
        `field` : FieldInterface
        
        `boundary` : str
        
        `value` : float = 0.0
        
        `nghosts` : int = 1 

        Returns
        --------------------------------
        """
        return super().apply(field, value)
    

    def __str__(self):
        return "NoSlip"
    


# TODO To be implemented
class Sponge(BoundaryCondition):
    def __init__(self, boundary, value:float|np.ndarray, flow_type:str='zerograd', nghosts:int=1):
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)

class Damped(BoundaryCondition):
    def __init__(self, boundary, value:float|np.ndarray, flow_type:str='zerograd', nghosts:int=1):
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)

class ConvectiveInOut(BoundaryCondition):
    """
    ∂ϕ/∂t + U_c(∂ϕ/∂x) = 0
    where U_c is the convection velocity
    """
    def __init__(self, boundary, value:float|np.ndarray, flow_type:str='zerograd', nghosts:int=1):
        if flow_type is None: flow_type = 'zerograd'
        super().__init__(boundary, value, flow_type, nghosts)