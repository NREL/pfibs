"""Solves the lid-driven cavity Navier-Stokes equation with default block structure and PC."""

## Future-proofing for Python3+
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
from dolfin_adjoint import *
from pfibs import *

## Generate mesh ##
mesh = RectangleMesh(Point(0.0,0.0),Point(1.0,1.0),64,64)

## Define boundaries subdomains ##
class Lid(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],1.0)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],0.0) or near(x[0]*(1.0-x[0]),0)

## Create MeshFunction to mark the boundaries ##
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
Lid().mark(boundaries, 1)
Walls().mark(boundaries, 2)

## Setup Taylor Hood Function Space ##
CG2 = VectorElement("CG", mesh.ufl_cell(), 2)
CG1 = FiniteElement("CG", mesh.ufl_cell(), 1)
W = MixedElement([CG2,CG1])
W = FunctionSpace(mesh, W )

## Create Trial, Test, and Solutions Functions ##
U  = TrialFunction(W)
V  = TestFunction(W)
X  = Function(W)
u_, p_   = split(U)
v, q   = split(V)
u, p = split(X)

## Set up No-slip BC on the walls ##
bcs = []
zero = Constant((0.0,0.0))
bcs.append( DirichletBC(W.sub(0), zero, boundaries, 2) )

## Set Lid-driven flow BC ##
inflow = Constant((1.0,0.0))
bcs.append( DirichletBC(W.sub(0), inflow, boundaries, 1) )

## Define the forcing function ##
f = Constant((0.0,-1.0))

## Define Viscosity and stabilization term ##
nu = Constant(1.0)
stab = mesh.hmin()

## Variational form for block solver ##
F =  nu*inner(grad(u), grad(v))*dx \
        + inner(dot(grad(u), u), v)*dx \
        - p*div(v)*dx - div(u)*q*dx \
        - inner(f,v)*dx + stab*p*q*dx

## Calculate Jacobian ##
J = derivative(F,X)

## Setup block problem ##
problem = BlockProblem(J, F, X, bcs=bcs)

## Add fields
problem.field('v',0,solver={'ksp_type':'preonly','pc_type':'lu'})
problem.field('p',1,solver={'ksp_type':'preonly','pc_type':'lu'})

## Setup block solver ##
solver = NonlinearBlockSolver(problem)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])

G = assemble(dot(u,u)*dx)
h = Constant(0.1)
G_hat = ReducedFunctional(G, Control(nu))
conv_rate = taylor_test(G_hat, nu, h)
