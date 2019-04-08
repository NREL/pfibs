# -*- coding: utf-8 -*-

# Parallel FEniCS Implementation of Block Solvers (pFibs) Copyright (c) 2018 Alliance for Sustainable Energy, LLC. 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the 
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
#    disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. The name of the copyright holder(s), any contributors, the United States Government, the United States Department 
#    of Energy, or any of their employees may not be used to endorse or promote products derived from this software 
#    without specific prior written permission from the respective party.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND ANY CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER(S), ANY CONTRIBUTORS, THE UNITED STATES GOVERNMENT, 
# OR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Error message handling for PETSc
from dolfin import SubSystemsManager
SubSystemsManager.init_petsc()
from petsc4py import PETSc
PETSc.Sys.pushErrorHandler("traceback")
del SubSystemsManager, PETSc

# Import public API
from pfibs.block_preconditioners import *
from pfibs.custom_linear import CustomKrylovSolver
from pfibs.custom_nonlinear import NLP, NS
from pfibs.block_problem import BlockProblem
from pfibs.block_solver import LinearBlockSolver, NonlinearBlockSolver
