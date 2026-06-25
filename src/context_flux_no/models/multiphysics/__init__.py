from .abstract import AbstractMultiphysicsOperator as AbstractMultiphysicsOperator
from .disco import DISCO as DISCO
from .dpot import DPOT as DPOT
from .fno import NaiveFNO as NaiveFNO, SpatiotemporalFNO as SpatiotemporalFNO
from .hyperfluxfno import (
    HyperFluxFNO as HyperFluxFNO,
    HyperFluxFNOLocal as HyperFluxFNOLocal,
)
from .wrapper import MultiphysicsWrapper as MultiphysicsWrapper
