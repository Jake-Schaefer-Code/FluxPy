import numpy as np
from abc import abstractmethod, ABC


__all__ = ["FieldInterface", "MeshInterface", "BoundaryInterface"]

class FieldInterface(ABC):
    @abstractmethod
    def __getitem__(self, stencil) -> np.ndarray:
        pass

    @abstractmethod
    def __setitem__(self, stencil, value):
        pass


class MeshInterface(ABC):
    nghosts:int
    ncells:int
    dx:float
    dy:float
    priority_map:dict
    @abstractmethod
    def __getitem__(self, prim:str) -> FieldInterface:
        pass

    @abstractmethod
    def __setitem__(self, stencil, value):
        pass

class BoundaryInterface(ABC):
    @abstractmethod
    def apply(self, field:FieldInterface, boundary:str, value:np.ndarray, nghosts:int=1, *args):
        pass

    @abstractmethod
    def __str__(self):
        pass