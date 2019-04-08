"""Solves the lid-driven cavity Navier-Stokes equation with PCD schur complement."""

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

## Define Viscosity ##
nu = Constant(1.0)

## Variational form for block solver ##
F =  nu*inner(grad(u), grad(v))*dx \
        + inner(dot(grad(u), u), v)*dx \
        - p*div(v)*dx - div(u)*q*dx \
        -inner(f,v)*dx

## Calculate Jacobian ##
J = derivative(F,X)

## Define the three operators and the boundary condition for the pressure preconditioner ##
bc_pcd1 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 1 )

## Setup block problem ##
block_structure = [['u',[[0,0],[0,1]]],['p',1]]
problem = BlockProblem(J, F, X, bcs=bcs, block_structure=block_structure)

## Built-in function calls ##
problem.field('u',[[0,0],[0,1]],solver={
    'ksp_type':'gmres',
    'pc_type':'hypre',
})
problem.field('p',1,solver={
    'ksp_type':'gmres',
    'pc_type':'python',
    'pc_python_type':'pfibs.PCDPC',
})
problem.split('s1',['u','p'],solver={
    'ksp_type':'fgmres',
    'pc_fieldsplit_type':'schur',
    'pc_fieldsplit_schur_precondition':'user',
    'ksp_monitor_true_residual': True,
})

## PCDPC context ##
ctx = {
    'nu': 1.0,
    'vp_spaces': [0,1],
    'bcs_aP': bc_pcd1,
    'solver': {},
}

## Setup block solver ##
solver = NonlinearBlockSolver(problem,ctx=ctx)

## Solve problem ##
timer = Timer("Solve Problem")
solver.solve()
timer.stop()

list_timings(TimingClear.keep, [TimingType.wall])
