"""Solves a mixed-poisson problem using schur complement approach
implemented via PETSc, solved using PythonPC preconditioner"""
# -*- coding: utf-8 -*-

## Future-proofing for Python3+ ##
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
from pfibs import *

## Create mesh ##
mesh = UnitSquareMesh(160,160)
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
class LeftRight(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
G = BoundarySource(mesh, degree=2)
bc = DirichletBC(W.sub(0), G, boundary)

boundaries = MeshFunction('size_t',mesh,1,3)
leftright = LeftRight()
leftright.mark(boundaries,1)
ds = Measure("ds",subdomain_data=boundaries)

## Interior Penalty for schur complement matrix ##
n = FacetNormal(W)
alpha = Constant(4.0)
gamma = Constant(8.0)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))*0.5
aP = -dot(grad(q),grad(p))*dx \
    + dot(avg(grad(q)), jump(p,n))*dS \
    + dot(jump(q,n), avg(grad(p)))*dS \
    - alpha/h_avg * dot(jump(q, n), jump(p, n))*dS \
    + dot(grad(q), p*n)*ds(1) \
    + dot(q*n, grad(p))*ds(1) \
    - (gamma/h)*q*p*ds(1)

## Setup block problem ##
problem = BlockProblem(a,L,w,bcs=bc)
problem.field('u',0,solver={
    'ksp_type':'preonly',
    'pc_type':'bjacobi'
})
problem.field('p',1,solver={
    'ksp_type':'preonly',
    'pc_type':'python',
    'pc_python_type':'pfibs.PythonPC'
})
problem.split('s1',['u','p'],solver={
    'ksp_type':'gmres',
    'pc_fieldsplit_type':'schur',
    'pc_fieldsplit_schur_fact_type':'upper',
    'pc_fieldsplit_schur_precondition':'user',
    'ksp_monitor_true_residual':True
})

## PythonPC context ##
ctx = {
    'aP': aP,
    'solver': {'pc_type': 'hypre'}
}

## Setup block solver ##
solver = LinearBlockSolver(problem, ctx=ctx)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])
