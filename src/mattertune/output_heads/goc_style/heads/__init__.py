from typing import Annotated, TypeAlias
from pydantic import Field

from .force_direct import DirectForceOutputHeadConfig as DirectForceOutputHeadConfig
from .force_direct import DirectForceOutputHead as DirectForceOutputHead
from .force_gradient import GradientForceOutputHeadConfig as GradientForceOutputHeadConfig
from .force_gradient import GradientForceOutputHead as GradientForceOutputHead
from .scaler_direct import DirectScalerOutputHeadConfig as DirectScalerOutputHeadConfig
from .scaler_direct import DirectScalerOutputHead as DirectScalerOutputHead
from .scaler_referenced import ReferencedScalerOutputHeadConfig as ReferencedScalerOutputHeadConfig
from .scaler_referenced import ReferencedScalerOutputHead as ReferencedScalerOutputHead
from .stress_direct import DirectStressOutputHeadConfig as DirectStressOutputHeadConfig
from .stress_direct import DirectStressOutputHead as DirectStressOutputHead
from .stress_gradient import GradientStressOutputHeadConfig as GradientStressOutputHeadConfig
from .stress_gradient import GradientStressOutputHead as GradientStressOutputHead
from .global_direct import GlobalScalerOutputHeadConfig as GlobalScalerOutputHeadConfig
from .global_direct import GlobalScalerOutputHead as GlobalScalerOutputHead
from .global_direct import GlobalBinaryClassificationOutputHeadConfig as GlobalBinaryClassificationOutputHeadConfig
from .global_direct import GlobalBinaryClassificationOutputHead as GlobalBinaryClassificationOutputHead
from .global_direct import GlobalMultiClassificationOutputHeadConfig as GlobalMultiClassificationOutputHeadConfig
from .global_direct import GlobalMultiClassificationOutputHead as GlobalMultiClassificationOutputHead

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