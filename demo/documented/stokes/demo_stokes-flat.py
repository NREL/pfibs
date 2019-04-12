"""Solves the lid-driven cavity Stokes equation using the chur complement. 
Implemented via PETSc and provides Custom PC matrix."""

## Future-proofing for Python3+
from __future__ import print_function

## Import preliminaries ##
from dolfin import *
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
CG2 = FiniteElement('CG', mesh.ufl_cell(), 2)
CG1 = FiniteElement("CG", mesh.ufl_cell(), 1)
W = MixedElement([CG2,CG2,CG1])
W = FunctionSpace(mesh, W)

## Create Trial, Test, and Solutions Functions ##
U  = TrialFunction(W)
V  = TestFunction(W)
X  = Function(W)
u0, u1, p   = split(U)
v0, v1, q   = split(V)
u = as_vector([u0, u1])
v = as_vector([v0, v1])

## Set up No-slip BC on the walls ##
bcs = []
zero = Constant(0.0)
bcs.append( DirichletBC(W.sub(0), zero, boundaries, 2) )
bcs.append( DirichletBC(W.sub(1), zero, boundaries, 2) )

## Set Lid-driven flow BC ##
inflow = Constant(1.0)
bcs.append( DirichletBC(W.sub(0), inflow, boundaries, 1) )
bcs.append( DirichletBC(W.sub(1), zero, boundaries, 1) )

## Define the forcing function ##
f = Constant((0.0,-1.0))

## Define Viscosity term ##
nu = Constant(1.0)

## Variational form for block solver ##
a =  nu*inner(grad(u), grad(v))*dx \
        - p*div(v)*dx - div(u)*q*dx
L = inner(f,v)*dx

## Custom PC for schur complement ##
aP = nu*inner(grad(u), grad(v))*dx - 1/nu*p*q*dx

## Setup block problem ##
problem = BlockProblem(a, L, X, bcs=bcs, aP=aP)
problem.field('u', [0, 1],solver={
    'ksp_type':'preonly',
    'pc_type':'hypre'
})
problem.field('p',2,solver={
    'ksp_type':'preonly',
    'pc_type':'bjacobi',
})
problem.split('s1',['u','p'],solver={
    'ksp_type':'gmres',
    'pc_fieldsplit_type':'schur',
    'pc_fieldsplit_off_diag_use_amat': True,
    'ksp_monitor_true_residual': True
})

## Setup block solver ##
solver = LinearBlockSolver(problem)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])
