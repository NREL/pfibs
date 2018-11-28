"""Solves the (Navier-)Stokes equation with user defined block_structure.
Compares block solver solution with builtin FEniCS solver."""
# -*- coding: utf-8 -*-

### Future-proofing for Python3+
from __future__ import print_function

### IM_port dolfin and nuM_py and time ###
from dolfin import *
from pfibs import *
import os

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
CG2 = FiniteElement("CG", mesh.ufl_cell(), 2)
CG1 = FiniteElement("CG", mesh.ufl_cell(), 1)
W = MixedElement([CG2,CG2,CG1])
W = FunctionSpace(mesh, W )

### Create Trial, Test, and Solutions Functions ###
U  = TrialFunction(W)
V  = TestFunction(W)
X1  = Function(W)
X2  = Function(W)
ux, uy, p   = split(U)
vx, vy, q   = split(V)
u = as_vector((ux, uy))
v = as_vector((vx, vy))

### Set up No-slip BC on the walls ###
bcs = []
bcs.append( DirichletBC(W.sub(0), 0.0, boundaries, 1) )
bcs.append( DirichletBC(W.sub(0), 0.0, boundaries, 2) )

### Set In-Flow BC ###
bcs.append( DirichletBC(W.sub(0), 0.0, boundaries, 4) )
bcs.append( DirichletBC(W.sub(1), 4.0, boundaries, 4) )

### Define the wind ###
w = Constant((2.0,1.0))
# w = Expression(("2*x[1]*(1-x[0]*x[0])","-2*x[0]*(1-x[1]*x[0])"),degree=2)

### Define the forcing function ###
f = Constant((-10.0,0.0))

### Define Viscosity ###
nu = 1.0

### Define Variational Form at the previous Newton Step: U_k ###
F =  nu*inner(grad(u), grad(v))*dx + inner(dot(grad(u), w), v)*dx - p*div(v)*dx - div(u)*q*dx

### Add in the force ###
F += -inner(f,v)*dx

### Split F for linear solves ###
a = lhs(F)
L = rhs(F)

### Define the block structure ###
# block_structure = None
# block_structure = [['u',[0,1]],['p',[2]]]
block_structure = [['ux',[0]],['uy',[1]],['p',[2]]]
# block_structure = [['ux',0],['uy',1],['p',2]]


# dict_test = dict(block_structure)
# for i in range(len(block_structure)):
#     dict_test.update( {i+3:block_structure[i][1]} )
# print(dict_test)
# print(dict_test[4,'u'])
# exit()

### Build the linear block solver ###
problem = BlockProblem(a, L, X1,bcs=bcs, block_structure=block_structure)

### Define the three operators and the boundary condition for the pressure preconditioner ###
M_p = Constant(1.0/nu)*p*q*dx
K_p = Constant(1.0/nu)*dot(grad(p), w)*q*dx
A_p = inner(grad(p), grad(q))*dx
bc_pcd1 = DirichletBC(W.sub(1), 0.0, boundaries, 4)

### Build a precondition and set it for the pressure block ###
preconditoner = PCDPC_BRM1(M_p,K_p,A_p,bc_pcd1)

### Set the solver/preconditioners for the blocks ###
problem.KSPType('fgmres')
problem.SubKSPType('ux','cg')
problem.SubKSPType('uy','cg')
problem.SubPCType('ux','hypre')
problem.SubPCType('uy','hypre')
problem.SubPCType('p',preconditoner)

### Set some ksp options ###
PETScOptions.set('ksp_monitor_true_residual')

solver = LinearBlockSolver(problem)

### Solve the system ###
solver.solve()
solve(a == L, X2, bcs)

### Split Solution ###
ux1, uy1, p1 = X1.split()
ux2, uy2, p2 = X2.split()

### Save solutions ###
File("output/"+os.path.splitext(__file__)[0]+"/block/ux_block.pvd") << ux1
File("output/"+os.path.splitext(__file__)[0]+"/block/uy_block.pvd") << uy1
File("output/"+os.path.splitext(__file__)[0]+"/block/p_block.pvd") << p1
File("output/"+os.path.splitext(__file__)[0]+"/exact/ux_exact.pvd") << ux2
File("output/"+os.path.splitext(__file__)[0]+"/exact/uy_exact.pvd") << uy2
File("output/"+os.path.splitext(__file__)[0]+"/exact/p_exact.pvd") << p2

### Plot solution ###
# plt.figure()
# pp = plot(u1[1],title="block")
# plt.colorbar(pp)

# plt.figure()
# pp = plot(u2[1],title="exact")
# plt.colorbar(pp)

plt.figure()
pp = plot(uy1-uy2,title="difference")
plt.colorbar(pp)
plt.show()
