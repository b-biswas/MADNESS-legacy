"""init file."""
from importlib import metadata

__version__ = metadata.version("maddeb")

from . import (
    batch_generator,
    boxplot,
    callbacks,
    dataset_generator,
    Deblender,
    extraction,
    FlowVAEnet,
    losses,
    metrics,
    model,
    utils,
)
