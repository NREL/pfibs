"""Solves a mixed-poisson problem using schur complement approach
implemented via built-in function calls to PETSc"""

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

## Built-in function calls ##
problem.field('u',0,solver={
    'ksp_type':'preonly',
    'pc_type':'bjacobi'
})
problem.field('p',1,solver={
    'ksp_type':'preonly',
    'pc_type':'hypre'
})
problem.split('s1',['u','p'],solver={
    'ksp_type':'gmres',
    'pc_fieldsplit_type':'schur',
    'pc_fieldsplit_schur_fact_type':'upper',
    'pc_fieldsplit_schur_precondition':'selfp',
    'ksp_monitor_true_residual':True
})

## Setup block solver ##
solver = LinearBlockSolver(problem)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])
