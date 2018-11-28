"""Solves the lid-driven cavity Navier-Stokes equation with user defined block_structure.
Compares block solver solution with builtin FEniCS solver."""
# -*- coding: utf-8 -*-

## Future-proofing for Python3+
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
from pfibs import *
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import argparse, os, sys
from petsc4py import PETSc
rank = PETSc.COMM_WORLD.Get_rank()

## Parse input arguments ##
parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ksp_monitor", dest="kspmonitor", action="store_true",
                    help="Show the KSP monitor")
parser.add_argument("-nu", type=float, dest="nu", default=1.0,
                    help="Viscosity")
parser.add_argument("-nx", type=int, dest="nx", default=64,
                    help="Number of elements in the x direction")
parser.add_argument("-ny", type=int, dest="ny", default=64,
                    help="Number of elements in the y direction")
parser.add_argument("-plot_view", dest="plotview", action="store_true",
                    help="Show the matplotlib figures")
parser.add_argument("-save_output", dest="saveoutput", action="store_true",
                    help="Save the solution output to a pvd file")
parser.add_argument("-snes_view", dest="snesview", action="store_true",
                    help="Show the SNES details")
args = parser.parse_args(sys.argv[1:])

## Create output directory if needed ##
if args.saveoutput and not rank:
    if not os.path.exists("output/"+os.path.splitext(__file__)[0]+"/block/"):
        os.makedirs("output/"+os.path.splitext(__file__)[0]+"/block/")
    if not os.path.exists("output/"+os.path.splitext(__file__)[0]+"/exact/"):
        os.makedirs("output/"+os.path.splitext(__file__)[0]+"/exact/")

## Generate mesh ##
Lx = 1.0
Ly = 1.0
nx = args.nx
ny = args.ny
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),nx,ny)

## Define boundaries subdomains ##
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],Ly)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],0.0)

## Create MeshFunction to mark the boundaries ##
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Top().mark(boundaries, 3)
Bottom().mark(boundaries, 4)

## Setup Taylor Hood Function Space ##
CG2 = FiniteElement("CG", mesh.ufl_cell(), 2)
CG1 = FiniteElement("CG", mesh.ufl_cell(), 1)
W = MixedElement([CG2,CG2,CG1])
W = FunctionSpace(mesh, W )

## Create Trial, Test, and Solutions Functions ##
U  = TrialFunction(W)
V  = TestFunction(W)
X1  = Function(W)
X2  = Function(W)
ux_,uy_, p_   = split(U)
vx, vy, q   = split(V)
ux1, uy1, p1 = split(X1)
ux2, uy2, p2 = split(X2)
u1 = as_vector((ux1,uy1))
u2 = as_vector((ux2,uy2))
v = as_vector((vx,vy))

## Set up No-slip BC on the walls ##
zero = Constant(0.0)
bcs = []
bcs.append( DirichletBC(W.sub(0), zero, boundaries, 1) )
bcs.append( DirichletBC(W.sub(1), zero, boundaries, 1) )
bcs.append( DirichletBC(W.sub(0), zero, boundaries, 2) )
bcs.append( DirichletBC(W.sub(1), zero, boundaries, 2) )
bcs.append( DirichletBC(W.sub(0), zero, boundaries, 4) )
bcs.append( DirichletBC(W.sub(1), zero, boundaries, 4) )

## Set Lid-driven flow BC ##
inflow = Constant(1.0)
bcs.append( DirichletBC(W.sub(0), inflow, boundaries, 3) )
bcs.append( DirichletBC(W.sub(1), zero, boundaries, 3) )

## Define the forcing function ##
f = Constant((0.0,-1.0))

## Define Viscosity ##
nu = Constant(args.nu)

## Variational form for block solver ##
F1 =  nu*inner(grad(u1), grad(v))*dx + inner(dot(grad(u1), u1), v)*dx - p1*div(v)*dx - div(u1)*q*dx

## Add in the force ##
F1 += -inner(f,v)*dx

## Variational form for default dolfin solver ##
F2 =  nu*inner(grad(u2), grad(v))*dx + inner(dot(grad(u2), u2), v)*dx - p2*div(v)*dx - div(u2)*q*dx

## Add in the force ##
F2 += -inner(f,v)*dx

## Calculate Jacobian ##
J1 = derivative(F1,X1)
J2 = derivative(F2,X2)

## Define the block structure ##
block_structure = [['ux',[0]],['uy',[1]],['p',[2]]]

## Build the nonlinear block solver ##
problem = BlockProblem(J1, F1, X1, bcs=bcs, block_structure=block_structure)

## Define the three operators and the boundary condition for the pressure preconditioner ##
M_p = Constant(1.0/nu)*p_*q*dx
K_p = Constant(1.0/nu)*dot(grad(p_), u1)*q*dx
A_p = inner(grad(p_), grad(q))*dx
bc_pcd1 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 3 )

## Build a precondition and set it for the pressure block ##
preconditoner = PCDPC_BRM1(M_p,K_p,A_p,bc_pcd1)

## Built-in function calls ##
problem.KSPType('fgmres')
problem.SubKSPType('ux','cg')
problem.SubKSPType('uy','cg')
problem.SubPCType('ux','hypre')
problem.SubPCType('uy','hypre')
problem.SubPCType('p',preconditoner)

## PETSc Command-line options ##
if args.kspmonitor:
    PETScOptions.set('ksp_monitor_true_residual')
if args.snesview:
    PETScOptions.set('ksp_view')

solver = NonlinearBlockSolver(problem)

## Solve the block system ##
solver.solve()

# Solve the normal system ##
solve(F2 == 0, X2, bcs=bcs)
ux1, uy1, p1 = X1.split()
ux2, uy2, p2 = X2.split()

## Save output ot pvd files ##
if args.saveoutput:
    File("output/"+os.path.splitext(__file__)[0]+"/block/ux_block.pvd") << ux1
    File("output/"+os.path.splitext(__file__)[0]+"/block/uy_block.pvd") << uy1
    File("output/"+os.path.splitext(__file__)[0]+"/block/p_block.pvd") << p1
    File("output/"+os.path.splitext(__file__)[0]+"/exact/ux_exact.pvd") << ux2
    File("output/"+os.path.splitext(__file__)[0]+"/exact/ux_exact.pvd") << uy2
    File("output/"+os.path.splitext(__file__)[0]+"/exact/p_exact.pvd") << p2

## Plot matplotlib ##
if args.plotview:
    plt.figure()
    pp = plot(u1[1],title="block")
    plt.colorbar(pp)

    plt.figure()
    pp = plot(u2[1],title="exact")
    plt.colorbar(pp)

    plt.figure()
    pp = plot(u1[1]-u2[1],title="difference")
    plt.colorbar(pp)
    plt.show()
