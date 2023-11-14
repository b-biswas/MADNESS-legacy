"""init file."""
from importlib import metadata

__version__ = metadata.version("maddeb")

from . import (
    Deblender,
    FlowVAEnet,
    batch_generator,
    boxplot,
    callbacks,
    dataset_generator,
    extraction,
    losses,
    metrics,
    model,
    utils,
)
