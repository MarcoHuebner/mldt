"""Optional native-extension declaration; project metadata is in pyproject.toml."""

import os

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = []
if os.environ.get("MLDT_DISABLE_NATIVE", "0") not in {"1", "true", "TRUE"}:
    extensions = cythonize(
        [Extension("mldt._native.core", ["mldt/_native/core.pyx"], optional=True)]
    )

setup(ext_modules=extensions)
