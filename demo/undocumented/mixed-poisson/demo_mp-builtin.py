"""Solves a mixed-poisson problem using schur complement approach
implemented via built-in function calls"""
# -*- coding: utf-8 -*-

## Future-proofing for Python3+ ##
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
from pfibs import *
import matplotlib.pyplot as plt
import argparse, os, sys
from petsc4py import PETSc
rank = PETSc.COMM_WORLD.Get_rank()

## Parse input arguments ##
parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ksp_view", dest="kspview", action="store_true",
                    help="Show the KSP details")
parser.add_argument("-nx", type=int, dest="nx", default=40,
                    help="Number of elements in the x direction")
parser.add_argument("-ny", type=int, dest="ny", default=40,
                    help="Number of elements in the y direction")
parser.add_argument("-plot_view", dest="plotview", action="store_true",
                    help="Show the matplotlib figures")
parser.add_argument("-save_output", dest="saveoutput", action="store_true",
                    help="Save the solution output to a pvd file")
args = parser.parse_args(sys.argv[1:])

## Create output directory if needed ##
if args.saveoutput and not rank:
    if not os.path.exists("output/"+os.path.splitext(__file__)[0]+"/"):
        os.makedirs("output/"+os.path.splitext(__file__)[0]+"/")

## Create mesh ##
mesh = UnitSquareMesh(args.nx,args.ny)
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
problem.SchurType('selfp')
problem.SubKSPType('u','preonly')
problem.SubKSPType('p','preonly')
problem.SubPCType('u','bjacobi')
problem.SubPCType('p','hypre')

## PETSc Command-line options ##
PETScOptions.set('ksp_monitor_true_residual')
PETScOptions.set('ksp_converged_reason')
if args.kspview:
    PETScOptions.set('ksp_view')

## Setup block solver ##
solver = LinearBlockSolver(problem)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()
(vel, pres) = w.split()

# Save output to pvd files ##
if args.saveoutput:
    File("output/"+os.path.splitext(__file__)[0]+"/u.pvd") << vel
    File("output/"+os.path.splitext(__file__)[0]+"/v.pvd") << pres

## Plot matplotlib ##
if args.plotview:
    plt.figure()
    plot(vel)
    plt.figure()
    plot(pres)
    plt.show()

list_timings(TimingClear.keep, [TimingType.wall])
