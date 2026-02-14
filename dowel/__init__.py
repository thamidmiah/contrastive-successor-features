"""Logger module.

This module instantiates a global logger singleton.
"""
from dowel.histogram import Histogram
from dowel.logger import Logger, LoggerWarning, LogOutput
from dowel.simple_outputs import StdOutput, TextOutput
from dowel.tabular_input import TabularInput
from dowel.csv_output import CsvOutput  # noqa: I100

# Make TensorBoard optional - don't fail if TensorFlow is slow/missing
try:
    from dowel.tensor_board_output import TensorBoardOutput
    _has_tensorboard = True
except (ImportError, Exception):
    TensorBoardOutput = None
    _has_tensorboard = False

logger = Logger()
tabular = TabularInput()

__all__ = [
    'Histogram',
    'Logger',
    'CsvOutput',
    'StdOutput',
    'TextOutput',
    'LogOutput',
    'LoggerWarning',
    'TabularInput',
    'TensorBoardOutput',
    'logger',
    'tabular',
]
