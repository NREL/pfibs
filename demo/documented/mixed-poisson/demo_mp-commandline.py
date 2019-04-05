"""Solves a mixed-poisson problem using schur complement approach
implemented via PETSc commandline options"""
# -*- coding: utf-8 -*-

## Future-proofing for Python3+ ##
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
from pfibs import *

## Create mesh ##
mesh = UnitSquareMesh(40,40)
V = FiniteElement("RT",mesh.ufl_cell(),1)
P = FiniteElement("DG",mesh.ufl_cell(),0)
W = FunctionSpace(mesh,V*P)
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
w = Function(W)

## Weak formulation ##
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
a = (dot(v,u) + div(v)*p + q*div(u))*dx
L = -q*f*dx

## Boundary conditions ##
class BoundarySource(UserExpression):
  def __init__(self, mesh, **kwargs):
    self.mesh = mesh
    super().__init__(**kwargs)
  def eval_cell(self, values, x, ufc_cell):
    cell = Cell(self.mesh, ufc_cell.index)
    n = cell.normal(ufc_cell.local_facet)
    g = sin(5*x[0])
    values[0] = g*n[0]
    values[1] = g*n[1]
  def value_shape(self):
    return (2,)
def boundary(x):
  return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
G = BoundarySource(mesh, degree=2)
bc = DirichletBC(W.sub(0), G, boundary)

## Setup block problem ##
problem = BlockProblem(a,L,w,bcs=bc)

## PETSc Command-line options ##
solver = {
    'ksp_type':'gmres',
    'pc_type':'fieldsplit',
    'pc_fieldsplit_type':'schur',
    'pc_fieldsplit_schur_fact_type':'upper',
    'pc_fieldsplit_schur_precondition':'selfp',
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'bjacobi',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'hypre',
    'ksp_monitor_true_residual': True,
    'ksp_converged_reason': True
}

## Setup block solver ##
solver = LinearBlockSolver(problem,solver=solver)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])
