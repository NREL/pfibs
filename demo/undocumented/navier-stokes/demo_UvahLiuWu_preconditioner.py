# DOE Project managers for Vechi, Wind, (VTOP, WETO)
# SUSAN HOFFMAN FORMAL CORRISPONDS REQUESTING DOE APPROVAL
# Byran cunningham
# Kathrine Dykes



# -*- coding: utf-8 -*-

### Future-proofing for Python3+
from __future__ import print_function

### IM_port dolfin and nuM_py and time ###
from dolfin import *
from pfibs import *
from petsc4py import PETSc
import numpy as np

### This iM_port iM_proves the plotter functionality on Mac ###
import matplotlib
matplotlib.use('TKAgg')

### IM_port matplotlib for plots ###
import matplotlib.pyplot as plt

### Set LogLevel depending on how much output is desired ###
set_log_level(LogLevel.INFO)

### Generate mesh ###
rf = 1
Lx = 1.0
Ly = 1.0
nx = 32*2**rf
ny = 32*2**rf
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),nx,ny)

### Define boundaries subdomains ###
class InFlow(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class OutFlow(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],Ly)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],0.0)

### Create MeshFunction to mark the boundaries ###
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
InFlow().mark(boundaries, 1)
OutFlow().mark(boundaries, 2)
Top().mark(boundaries, 3)
Bottom().mark(boundaries, 4)

### Export Mesh for Debuging ###
file = File("output/tests/mesh/boundaries.pvd")
file << boundaries

### Setup Taylor Hood Function Space ###
CG2 = VectorElement("CG", mesh.ufl_cell(), 2)
CG1 = FiniteElement("CG", mesh.ufl_cell(), 1)
W = MixedElement([CG2,CG1])
W = FunctionSpace(mesh, W )

### Create Trial, Test, and Solutions Functions ###
U  = TrialFunction(W)
V  = TestFunction(W)
X1  = Function(W)
X2  = Function(W)
u, p   = split(U)
v, q   = split(V)

### Set up No-slip BC on the walls ###
bcs = []
bcs.append( DirichletBC(W.sub(0).sub(0), (0.0), boundaries, 1) )
bcs.append( DirichletBC(W.sub(0).sub(0), (0.0), boundaries, 2) )
# bcs.append( DirichletBC(W.sub(0).sub(0), (1.0), boundaries, 3) )
# bcs.append( DirichletBC(W.sub(0).sub(0), (1.0), boundaries, 4) )
# bcs.append( DirichletBC(W.sub(0).sub(1), (2.0), boundaries, 1) )
# bcs.append( DirichletBC(W.sub(0).sub(1), (2.0), boundaries, 2) )
# bcs.append( DirichletBC(W.sub(0).sub(1), (2.0), boundaries, 3) )
# bcs.append( DirichletBC(W.sub(0).sub(1), (2.0), boundaries, 4) )
# bcs.append( DirichletBC(W.sub(1), (3.0), boundaries, 1) )
# bcs.append( DirichletBC(W.sub(1), (3.0), boundaries, 2) )
# bcs.append( DirichletBC(W.sub(1), (3.0), boundaries, 3) )
# bcs.append( DirichletBC(W.sub(1), (3.0), boundaries, 4) )

### Set In-Flow BC ###
inflow = Constant((0.0,4.0))
bcs.append( DirichletBC(W.sub(0), inflow, boundaries, 4) )

### Define the wind ###
w = Constant((2.0,1.0))
# w = Expression(("2*x[1]*(1-x[0]*x[0])","-2*x[0]*(1-x[1]*x[0])"),degree=2)

### Define the forcing function ###
f = Constant((-10.0,0.0))

### Define Viscosity ###
nu = 1.0e-0

### Define Variational Form at the previous Newton Step: U_k ###
F =  nu*inner(grad(u), grad(v))*dx + inner(dot(grad(u), w), v)*dx - p*div(v)*dx - div(u)*q*dx + 0.0002*p*q*dx

### Add in the force ###
F += -inner(f,v)*dx



### Split F for linear solves ###
a = lhs(F)
L = rhs(F)



test = Function(W)
test.vector()[W.sub(0).sub(0).dofmap().dofs()] = np.multiply(W.sub(0).sub(0).dofmap().dofs(),1)
test.vector()[W.sub(0).sub(1).dofmap().dofs()] = np.multiply(W.sub(0).sub(1).dofmap().dofs(),10)
test.vector()[W.sub(1).dofmap().dofs()] = np.multiply(W.sub(1).dofmap().dofs(),100)

# test = test.vector().get_local()
# X1.assign(test)
# for bc in bcs:
#     bc.apply(X1.vector())
### Build the linear block solver ###
# block_structure = [['all',[0,1]]]
problem = BlockProblem(a, L, X1, bcs=bcs)#, block_structure=block_structure)

### Define the three operators and the boundary condition for the pressure preconditioner ###
hval = float(mesh.hmin())
H_p = Constant(nu)*inner(grad(u), grad(v))*dx
K_p = inner(dot(grad(u), w), v)*dx - p*div(v)*dx - div(u)*q*dx + 0.0002*p*q*dx
rho_p = 1.0e-0

### Build a precondition and set it for the pressure block ###
preconditioner = UvahLiuWu(H_p,K_p,rho_p)

problem.KSPType('gmres')
# problem.PCType('lu')
problem.PCType(preconditioner)

### Set some ksp options ###
PETScOptions.set('ksp_monitor_true_residual')
PETScOptions.set('pc_factor_mat_solver_type','mumps')
PETScOptions.set("ksp_gmres_restart", 1250)
# PETScOptions.set("ksp_max_it", 15000)
# PETScOptions.set("ksp_rtol", 1.0e-02)
# PETScOptions.set("ksp_atol", 7e-0)
# PETScOptions.set('ksp_view')

solver = LinearBlockSolver(problem)

### Solve the system ###
solver.solve()
solve(a == L, X2, bcs)

### Split Solution ###
u1, p1 = X1.split()
u2, p2 = X2.split()

### Save solutions ###
File("output/tests/solution/block/u_block.pvd") << u1
File("output/tests/solution/block/p_block.pvd") << p1
File("output/tests/solution/exact/u_exact.pvd") << u2
File("output/tests/solution/exact/p_exact.pvd") << p2

### Plot solution ###
# plt.figure()
# pp = plot(u1[0],title="block")
# plt.colorbar(pp)
plt.figure()
pp = plot(u1[1],title="block")
plt.colorbar(pp)
# plt.figure()
# pp = plot(p1,title="block")
# plt.colorbar(pp)
# plt.show()

plt.figure()
pp = plot(u2[1],title="exact")
plt.colorbar(pp)

plt.figure()
pp = plot(u1[1]-u2[1],title="difference")
plt.colorbar(pp)

plt.figure()
pp = plot(u1[1]/u2[1],title="scale")
plt.colorbar(pp)
plt.show()





# plt.figure()
# pp = plot(u2[0],title="block")
# plt.colorbar(pp)
# plt.figure()
# pp = plot(u2[1],title="block")
# plt.colorbar(pp)
# plt.figure()
# pp = plot(p2,title="block")
# plt.colorbar(pp)
# plt.show()