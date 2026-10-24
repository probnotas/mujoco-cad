"""OpenCascade's STEP unit setting is global, and leaks.

``exportAssembly(..., unit="M")`` sets ``xstep.cascade.unit`` and never puts it
back, so a millimetre file written or read afterwards in the same process comes
out scaled by 1000.  Every STEP read and write in sim2cad goes through this
context manager so the setting is always explicit and always restored.
"""

from __future__ import annotations

import contextlib

__all__ = ["cascade_unit"]


@contextlib.contextmanager
def cascade_unit(unit: str):
    from OCP.Interface import Interface_Static

    key = "xstep.cascade.unit"
    previous = Interface_Static.CVal_s(key)
    Interface_Static.SetCVal_s(key, unit)
    try:
        yield
    finally:
        Interface_Static.SetCVal_s(key, previous or "MM")
