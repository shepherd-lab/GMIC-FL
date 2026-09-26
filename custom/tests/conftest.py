# Put the parent custom/ dir on sys.path so the tests can import the FL modules
# (bc_executor, fl_utils, model.*, data_loader.*, train.*, ...) and each other, whether
# run via `pytest custom/tests/` or a single `python custom/tests/test_X.py`.
import os
import sys

_CUSTOM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # custom/
if _CUSTOM not in sys.path:
    sys.path.insert(0, _CUSTOM)
