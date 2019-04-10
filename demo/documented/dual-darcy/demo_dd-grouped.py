"""Solves dual porosity/permeability darcy equations. Employs
nested fieldsplits with schur complements using builtin function calls."""
# -*- coding: utf-8 -*-

## Future-proofing for Python3+ ##
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
#import petsc4py
#petsc4py.init('-log_view')
from pfibs import *
from petsc4py import PETSc
import numpy as np

## Create mesh ##
mesh = UnitSquareMesh(40,40)
V = FiniteElement("RT",mesh.ufl_cell(),1)
P = FiniteElement("DG",mesh.ufl_cell(),0)
W = MixedElement([V,P,V,P])
W = FunctionSpace(mesh,W)
(u1,p1,u2,p2) = TrialFunctions(W)
(v1,q1,v2,q2) = TestFunctions(W)
w = Function(W)

## Boundary ##
class Left(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0],0.0)
class Right(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0],1.0)
class Bottom(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[1],0.0)
class Top(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[1],1.0)

## Mark boundaries ##
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

## Pressure boundary conditions ##
p1_left = Expression("1/pi*sin(pi*x[1]) - exp(3.316625*x[1])", degree=5)
p1_right = Expression("1/pi*exp(pi)*sin(pi*x[1]) - exp(3.316625*x[1])", degree=5)
p1_bottom = Expression("-1.0", degree=5)
p1_top = Expression("-27.5671484", degree=5)
p2_left = Expression("1/pi*sin(pi*x[1]) + 10*exp(3.316625*x[1])", degree=5)
p2_right = Expression("1/pi*exp(pi)*sin(pi*x[1]) + 10*exp(3.316625*x[1])", degree=5)
p2_bottom = Expression("10.0", degree=5)
p2_top = Expression("-275.671484", degree=5)

## Weak formulation ##
n = FacetNormal(mesh)
alpha1, alpha2 = Constant(1), Constant(10)
a = dot(v1, alpha1*u1)*dx + dot(v2, alpha2*u2)*dx \
    - div(v1)*p1*dx - div(v2)*p2*dx + q1*div(u1)*dx + q2*div(u2)*dx \
    + q1*(p1-p2)*dx - q2*(p1-p2)*dx
L = - dot(v1,n)*p1_left*ds(1) - dot(v2,n)*p2_left*ds(1) \
    - dot(v1,n)*p1_right*ds(2) - dot(v2,n)*p2_right*ds(2) \
    - dot(v1,n)*p1_bottom*ds(3) - dot(v2,n)*p2_bottom*ds(3) \
    - dot(v1,n)*p1_top*ds(4) - dot(v2,n)*p2_top*ds(4)

## Setup block problem ##
params1 = {
    "ksp_type":"preonly",
    "pc_type":"bjacobi"
}
params2 = {
    "ksp_type":"preonly",
    "pc_type":"hypre",
}
schur = {
    "ksp_type":"gmres",
    "pc_fieldsplit_type":"schur",
    "pc_fieldsplit_schur_fact_type":"full",
    "pc_fieldsplit_schur_precondition":"selfp",
    "ksp_monitor_true_residual": True
}
problem = BlockProblem(a,L,w,bcs=[])
problem.field('v',[0,2],solver=params1)
problem.field('p',[1,3],solver=params2)
problem.split('s1',['v','p'],solver=schur)

## Setup block solver ##
solver = LinearBlockSolver(problem)
solver.solve()
