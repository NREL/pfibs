# -*- coding: utf-8 -*-

# Copyright (C) 2019 NREL
# Other licensing stuff, blah blah blah


# Error message handling for PETSc
from dolfin import SubSystemsManager
SubSystemsManager.init_petsc()
from petsc4py import PETSc
PETSc.Sys.pushErrorHandler("traceback")
del SubSystemsManager, PETSc

# Import public API
from pfibs.block_preconditioners import PythonPC, PCD_BRM1, MyPCD, UvahLiuWu, Elman, Pre_Laplace
from pfibs.custom_linear import CustomKrylovSolver
from pfibs.custom_nonlinear import NLP, NS
from pfibs.block_problem import BlockProblem
from pfibs.block_solver import LinearBlockSolver, NonlinearBlockSolver
