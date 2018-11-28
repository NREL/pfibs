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
block_structure = [['u',[0]],['p',[1]]]
problem = BlockProblem(a,L,w,bcs=bc,block_structure=block_structure)

## PETSc Command-line options ##
PETScOptions.set('ksp_type','gmres')
PETScOptions.set('pc_type','fieldsplit')
PETScOptions.set('pc_fieldsplit_type','schur')
PETScOptions.set('pc_fieldsplit_schur_fact_type','upper')
PETScOptions.set('pc_fieldsplit_schur_precondition','selfp')
PETScOptions.set('fieldsplit_u_ksp_type', 'preonly')
PETScOptions.set('fieldsplit_u_pc_type', 'bjacobi')
PETScOptions.set('fieldsplit_p_ksp_type', 'preonly')
PETScOptions.set('fieldsplit_p_pc_type', 'hypre')
PETScOptions.set('ksp_monitor_true_residual')
PETScOptions.set('ksp_converged_reason')

## Setup block solver ##
solver = LinearBlockSolver(problem)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])
