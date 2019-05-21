#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("helper", ["pfibs/helper.pyx"])
]
setup(name ="pfibs",
      version="2018.1.0_0.5",
      author="Jeffery Allen, Justin Chang, Innokentiy Protasov",
      author_email="jallen@nrel.gov",
      url="https://github.com/NREL/pfibs",
      description="pFiBS: parallel FEniCS implementation of Block Solvers",
      packages=["pfibs","pfibs.block_preconditioners"],
      package_dir={"pfibs": "pfibs"},
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      include_dirs = [numpy.get_include()]
)
