from context_flux_no.models.multiphysics.hyperfluxfno.hyperfluxfno import (
    HyperNeuralOperator,
)

from .abstract import AbstractMultiphysicsOperator as AbstractMultiphysicsOperator
from .disco import DISCO as DISCO
from .dpot import DPOT as DPOT
from .fno import NaiveFNO as NaiveFNO, SpatiotemporalFNO as SpatiotemporalFNO
from .hyperfluxfno import (
    HyperFluxFNOLocal as HyperFluxFNOLocal,
    HyperNeuralOperator as HyperNeuralOperator,
)
from .wrapper import MultiphysicsWrapper as MultiphysicsWrapper
