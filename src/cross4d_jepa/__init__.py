"""Cross4D-JEPA: dense cross-modal correspondence distillation from a frozen 2D
foundation model into a 4D point-cloud encoder.

Public surface is intentionally small; import submodules directly:

    from cross4d_jepa.config import Config
    from cross4d_jepa.models import build_encoder
    from cross4d_jepa.engine.pretrain_dense import pretrain_dense
    from cross4d_jepa.engine.linear_probe import linear_probe
"""

__version__ = "1.0.0"
