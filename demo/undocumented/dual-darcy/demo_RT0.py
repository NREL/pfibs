"""Solves a dual porosity/permeability problem using nested
schur complements"""
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
C = FiniteElement("CG",mesh.ufl_cell(),1)
W = MixedElement([V,P,C,V,P,C])
W = FunctionSpace(mesh,W)
(u1,p1,c1,u2,p2,c2) = TrialFunctions(W)
(v1,q1,d1,v2,q2,d2) = TestFunctions(W)
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
    + q1*(p1-p2)*dx - q2*(p1-p2)*dx + inner(grad(c1),grad(d1))*dx \
    + inner(grad(c2),grad(d2))*dx
L = - dot(v1,n)*p1_left*ds(1) - dot(v2,n)*p2_left*ds(1) \
    - dot(v1,n)*p1_right*ds(2) - dot(v2,n)*p2_right*ds(2) \
    - dot(v1,n)*p1_bottom*ds(3) - dot(v2,n)*p2_bottom*ds(3) \
    - dot(v1,n)*p1_top*ds(4) - dot(v2,n)*p2_top*ds(4)




## Assemble system ##
#A = PETScMatrix()
#b = PETScVector()
#assemble_system(a,L,bcs=[],A_tensor=A,b_tensor=b)
#
### Extract FEniCS dof layout, global indices ##
#dof_total = np.array(W.dofmap().dofs())
#dof_v1 = np.array(W.sub(0).dofmap().dofs())
#dof_p1 = np.array(W.sub(1).dofmap().dofs())
#dof_t1 = np.array(W.sub(2).dofmap().dofs())
#dof_v2 = np.array(W.sub(3).dofmap().dofs())
#dof_p2 = np.array(W.sub(4).dofmap().dofs())
#dof_t2 = np.array(W.sub(5).dofmap().dofs())
#offset = np.min(dof_total)
#
### Create PetscSection ##
#section = PETSc.Section().create()
#section.setNumFields(6)
#section.setFieldName(0,'v1')
#section.setFieldName(1,'p1')
#section.setFieldName(2,'t1')
#section.setFieldName(3,'v2')
#section.setFieldName(4,'p2')
#section.setFieldName(5,'t2')
#section.setChart(0,len(dof_total))
#for i in np.nditer(dof_v1):
#  section.setDof(i-offset,1)
#  section.setFieldDof(i-offset,0,1)
#for i in np.nditer(dof_p1):
#  section.setDof(i-offset,1)
#  section.setFieldDof(i-offset,1,1)
#for i in np.nditer(dof_t1):
#  section.setDof(i-offset,1)
#  section.setFieldDof(i-offset,2,1)
#for i in np.nditer(dof_v2):
#  section.setDof(i-offset,1)
#  section.setFieldDof(i-offset,3,1)
#for i in np.nditer(dof_p2):
#  section.setDof(i-offset,1)
#  section.setFieldDof(i-offset,4,1)
#for i in np.nditer(dof_t2):
#  section.setDof(i-offset,1)
#  section.setFieldDof(i-offset,5,1)
#section.setUp()
#
### Create DM and assign PetscSection ##
#dm = PETSc.DMShell().create()
#dm.setDefaultSection(section)
#dm.setUp()
#
### Create KSP and assign DM ##
#ksp = PETSc.KSP().create()
#ksp.setDM(dm)
#ksp.setDMActive(False)
#PETScOptions.set('ksp_monitor_true_residual')
PETScOptions.set('ksp_view')
#PETScOptions.set('ksp_type', 'gmres')
#PETScOptions.set('pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_s2_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s2_pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_s2_pc_fieldsplit_type','additive')
#PETScOptions.set('fieldsplit_s2_fieldsplit_s1_pc_type', 'fieldsplit')
##PETScOptions.set('fieldsplit_s2_fieldsplit_s1_pc_fieldsplit_type', 'schur')
##PETScOptions.set('fieldsplit_s2_fieldsplit_s1_pc_fieldsplit_fact_type', 'full')
##PETScOptions.set('fieldsplit_s2_fieldsplit_s1_pc_fieldsplit_schur_precondition', 'selfp')
#PETScOptions.set('fieldsplit_s2_fieldsplit_s1_fieldsplit_0_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s2_fieldsplit_s1_fieldsplit_1_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s2_fieldsplit_s1_fieldsplit_0_pc_type', 'bjacobi')
#PETScOptions.set('fieldsplit_s2_fieldsplit_s1_fieldsplit_1_pc_type', 'hypre')
#PETScOptions.set('fieldsplit_s2_fieldsplit_2_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s2_fieldsplit_2_pc_type', 'hypre')
#PETScOptions.set('fieldsplit_s4_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s4_pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_s4_pc_fieldsplit_type','additive')
#PETScOptions.set('fieldsplit_s4_fieldsplit_s3_pc_type', 'fieldsplit')
##PETScOptions.set('fieldsplit_s4_fieldsplit_s3_pc_fieldsplit_type', 'schur')
##PETScOptions.set('fieldsplit_s4_fieldsplit_s3_pc_fieldsplit_fact_type', 'full')
##PETScOptions.set('fieldsplit_s4_fieldsplit_s3_pc_fieldsplit_schur_precondition', 'selfp')
#PETScOptions.set('fieldsplit_s4_fieldsplit_s3_fieldsplit_3_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s4_fieldsplit_s3_fieldsplit_4_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s4_fieldsplit_s3_fieldsplit_3_pc_type', 'bjacobi')
#PETScOptions.set('fieldsplit_s4_fieldsplit_s3_fieldsplit_4_pc_type', 'hypre')
#PETScOptions.set('fieldsplit_s4_fieldsplit_5_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_s4_fieldsplit_5_pc_type', 'hypre')
#PETScOptions.set('pc_fieldsplit_s2_fields', '0, 1, 2')
#PETScOptions.set('pc_fieldsplit_s4_fields', '3, 4, 5')
#PETScOptions.set('fieldsplit_s2_pc_fieldsplit_s1_fields', '0, 1')
#PETScOptions.set('fieldsplit_s4_pc_fieldsplit_s3_fields', '0, 1')
#PETScOptions.set('fieldsplit_s2_pc_fieldsplit_2_fields', '2')
#PETScOptions.set('fieldsplit_s4_pc_fieldsplit_5_fields', '2')
#
#### PETSc Command-line options ##
#PETScOptions.set('ksp_monitor_true_residual')
#PETScOptions.set('ksp_view')
#PETScOptions.set('ksp_type', 'gmres')
#PETScOptions.set('pc_type', 'fieldsplit')
#PETScOptions.set('pc_fieldsplit_0_fields', '0, 1, 2')
#PETScOptions.set('fieldsplit_0_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_0_pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_0_pc_fieldsplit_type','additive')
#PETScOptions.set('fieldsplit_0_pc_fieldsplit_0_fields', '0, 1')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_pc_fieldsplit_type', 'schur')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_pc_fieldsplit_fact_type', 'full')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_pc_fieldsplit_schur_precondition', 'selfp')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_fieldsplit_0_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_fieldsplit_1_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_fieldsplit_0_pc_type', 'bjacobi')
#PETScOptions.set('fieldsplit_0_fieldsplit_0_fieldsplit_1_pc_type', 'hypre')
#PETScOptions.set('fieldsplit_0_pc_fieldsplit_1_fields', '2')
#PETScOptions.set('fieldsplit_0_fieldsplit_2_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_0_fieldsplit_2_pc_type', 'hypre')
#PETScOptions.set('pc_fieldsplit_1_fields', '3, 4, 5')
#PETScOptions.set('fieldsplit_1_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_1_pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_1_pc_fieldsplit_type','additive')
#PETScOptions.set('fieldsplit_1_pc_fieldsplit_0_fields', '0, 1')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_pc_type', 'fieldsplit')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_pc_fieldsplit_type', 'schur')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_pc_fieldsplit_fact_type', 'full')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_pc_fieldsplit_schur_precondition', 'selfp')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_fieldsplit_3_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_fieldsplit_4_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_fieldsplit_3_pc_type', 'bjacobi')
#PETScOptions.set('fieldsplit_1_fieldsplit_0_fieldsplit_4_pc_type', 'hypre')
#PETScOptions.set('fieldsplit_1_pc_fieldsplit_1_fields', '2')
#PETScOptions.set('fieldsplit_1_fieldsplit_5_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_1_fieldsplit_5_pc_type', 'hypre')
#
### Solve ##
#ksp.setOperators(A.mat())
#ksp.setUp()
#ksp.setFromOptions()
#ksp.solve(b.vec(),w.vector().vec())

## Setup block problem ##
#block_structure = [['u1',[0]],['p1',[1]],['u2',[2]],['p2',[3]]]
params1 = {
    "ksp_type":"preonly",
    "pc_type":"bjacobi"
}
params2 = {
    "ksp_type":"gmres",
    "pc_type":"hypre",
    "ksp_monitor_true_residual": False
}
multi = {
    "ksp_type":"preonly",
    "pc_fieldsplit_type":"additive"
}
schur = {
    "ksp_type":"preonly",
    "pc_fieldsplit_type":"schur",
    "pc_fieldsplit_schur_fact_type":"full",
    "pc_fieldsplit_schur_precondition":"selfp"
}
problem = BlockProblem(a,L,w,bcs=[])
problem.add_field('0',0,solver=params1)
problem.add_field('1',1,solver=params2)
problem.add_field('2',2,solver=params2)
problem.add_field('3',3,solver=params1)
problem.add_field('4',4,solver=params2)
problem.add_field('5',5,solver=params2)
problem.add_split('s1',['0','1'],solver=schur)
problem.add_split('s2',['s1','2'],solver=multi)
problem.add_split('s3',['3','4'],solver=schur)
problem.add_split('s4',['s3','5'],solver=multi)
problem.add_split('s5',['s2','s4'],solver={"ksp_type":"fgmres"})
PETScOptions.set("ksp_monitor_true_residual")
PETScOptions.set("ksp_converged_reason")

## Setup block solver ##
solver = LinearBlockSolver(problem)
solver.solve()
