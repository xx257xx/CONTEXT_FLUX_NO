import grain
from jaxtyping import Array, Float


class SpatialDownsample(grain.transforms.Map):
    downsample: int

    def __init__(self, downsample: int = 1):
        super().__init__()
        self.downsample = downsample

    def map(
        self, x: Float[Array, "time channels *spatial_dims"]
    ) -> Float[Array, "time channels *spatial_dims_down"]:  # ty: ignore[invalid-method-override]
        n_spatial_dims = x.ndim - 2
        selection = [slice(None, None, self.downsample)] * n_spatial_dims
        return x[..., *selection]
