from typing import Annotated, TypeAlias
from pydantic import Field

from .force_direct import *
from .force_gradient import *
from .scaler_direct import *
from .scaler_referenced import *
from .stress_direct import *
from .stress_gradient import *
from .global_direct import *

GOCOutputHeadConfig: TypeAlias = Annotated[
    DirectForceOutputHeadConfig 
    | GradientForceOutputHeadConfig 
    | DirectScalerOutputHeadConfig 
    | ReferencedScalerOutputHeadConfig 
    | DirectStressOutputHeadConfig 
    | GradientStressOutputHeadConfig 
    | GlobalScalerOutputHeadConfig 
    | GlobalBinaryClassificationOutputHeadConfig 
    | GlobalMultiClassificationOutputHeadConfig,
    Field(description="head_name")
]