#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name ="pfibs",
      version="2018.1.0_0.1",
      author="Jeffery Allen, Justin Chang",
      author_email="jallen@nrel.gov",
      url="https://github.nrel.gov/jallen/block_solve",
      description="pFiBS: parallel FEniCS implementation of Block Solvers",
      packages=["pfibs"],
      package_dir={"pfibs": "pfibs"}
)
