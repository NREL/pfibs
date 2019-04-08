from dolfin import *
from pfibs import *
from math import asinh
from mpi4py import MPI
import sys,time

nx = int(sys.argv[1])

## Scaling ##
m3        = 1#1e6
m25       = 1#10*10**(2.5)
m2        = 1#1e4
m1        = 1#1e2
Lx        = 4.5e-5*m1


## Initialize problem ##
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(Lx,Lx,Lx), nx, nx, nx)

## Boundaries ##
class Left(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[0], 0.0)
class Right(SubDomain):
  def inside(self, x, on_boundary):
    return near(x[0], Lx)
left = Left()
right = Right()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)

## Domain measurements ##
dx = Measure("dx", domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
Gamma = assemble(Constant(1)*ds(1))
Omega = assemble(Constant(1)*dx)

## Function spaces ##
CG = FiniteElement("CG", mesh.ufl_cell(), 1)
W = MixedElement([CG,CG])
W = FunctionSpace(mesh, W)
V = FunctionSpace(mesh, CG)
U = TrialFunction(W)
V = TestFunction(W)
U_n = Function(W)
U_k = Function(W)
u, c = split(U)
v, d = split(V)


## Physical Parameters ##
tstep = 10
dt        = Constant(tstep)
hr_to_s   = 3600
R         = 8.3144621
T         = 298.15
Fday      = 96487.0
RTF       = R*T/Fday

## Electrolyte Parameters ##
ce_rest   = 1200/m3         # NOTE: Try various values e.g., 0.12, 1.2, 12, 120, and 1200
tp        = 0.465   
Ac        = 0               # Optionally increase this        
Ke        = 0.940/m1
De        = 1.2e-10*m2

## Anode ##
ifa       = 0.50
ca_max    = 9e9/m3
ka        = 0.00
aa        = 5.0e-01

## Cathode ##
ifc       = 0.30
cc_max    = 30000/m3
kc        = 0.2*m25
ac        = 5.0e-01

## Initial Concentrations ##
ca_0      = ifa*ca_max
ce_0      = ce_rest
cc_0      = ifc*cc_max

## Open circuit voltage ##
def OCP_c(x):
  a14=-3.640117692001490E+03;
  a13=1.317657544484270E+04;
  a12=- 1.455742062291360E+04;
  a11=- 1.571094264365090E+03;
  a10=+ 1.265630978512400E+04;
  a9=- 2.057808873526350E+03;
  a8=- 1.074374333186190E+04;
  a7=+ 8.698112755348720E+03;
  a6=- 8.297904604107030E+02;
  a5=- 2.073765547574810E+03;
  a4=+ 1.190223421193310E+03;
  a3=- 2.724851668445780E+02;
  a2=+ 2.723409218042130E+01;
  a1=- 4.158276603609060E+00;
  a0=5.314735633000300;
  addition= - 5.573191762723310E-04*exp(6.560240842659690E+00*(x**4.148209275061330E+01));
  OCP = (((((((((((((a14*x+a13)*x+a12)*x+a11)*x+a10)*x+a9)*x+a8)*x+a7)*x+a6)*x+a5)*x+a4)*x+a3)*x+a2)*x+a1)*x+a0+addition
  return (OCP)
def OCP_a(x):
  return 1.0e-9*x
Ua0 = OCP_a(ifa)
Uc0 = OCP_c(ifc)

## Exchange current density ##
def io_a(cs,ce,cs_max):
  #io = ka*sqrt((cs))*sqrt((ce))*(sqrt((cs_max-cs)))
  io = 100
  return(io)
def io_c(cs,ce,cs_max):
  io = kc*sqrt(abs(cs))*sqrt(abs(ce))*(sqrt(abs(cs_max-cs)))
  #io = kc*cs*ce*(cs_max-cs)#sqrt(abs(cs))*sqrt(abs(ce))*(sqrt(abs(cs_max-cs)))
  return(io)
i0a = io_a(ca_0,ce_0,ca_max)
i0c = io_c(cc_0,ce_0,cc_max)

## Flux ##
flux = -1.0*cc_max*Fday*Omega/(hr_to_s*Gamma)

## Initial potentials ##
ua_0 = 5.0
ue_0 = ua_0 - 2.0*RTF*asinh(Gamma*flux/(2.0*Gamma*i0a)) - Ua0
uc_0 = ue_0 - 2.0*RTF*asinh(Gamma*flux/(2.0*Gamma*i0c)) + Uc0

## Extract dofs for initial conditions ##
u_dofs = []
c_dofs = []
u_dofmap = W.sub(0).dofmap()
c_dofmap = W.sub(1).dofmap()
dof_min, dof_max = W.dofmap().ownership_range()
loc2global = W.dofmap().local_to_global_index
for cell in cells(mesh,"all"):
  if not cell.is_ghost():
    u_dofs.extend(u_dofmap.cell_dofs(cell.index()))
    c_dofs.extend(c_dofmap.cell_dofs(cell.index()))
u_dofs = list(filter(lambda dof: dof_min<=loc2global(dof)<dof_max,u_dofs))
c_dofs = list(filter(lambda dof: dof_min<=loc2global(dof)<dof_max,c_dofs))
all_dofs = [u_dofs,c_dofs]

## Initial conditions and guesses ##
all_0 = [ue_0,ce_0]
for i in range(2):
  U_k.vector()[all_dofs[i]] = all_0[i]
  U_n.vector()[all_dofs[i]] = all_0[i]

u_k, c_k = U_k.split(True)
u_n, c_n = U_n.split(True)

## Boundary conditions, not needed ##
def Node(x):
  return near(x[0],0.0) and near(x[1],Lx*0.5)
bcs = []
#bcs.append(DirichletBC(W.sub(0), ue_0, Node, "pointwise"))

## Electrolyte potential lux ##
def je(u,c):
  Ke_D = Constant((2*Ke*RTF)*(1.0-tp)*(1+Ac))
  return -Ke*grad(u) + Ke_D*grad(ln(c))

## Electrolyte concentration flux ##
def Ne(u,c):
  return -De*grad(c) + tp/Fday*je(u,c)

## Faraday Current function ##
def ise(us,ue,Us0,io):
  eta = (us - ue - Us0)
  return io*(exp(aa*eta/RTF)-exp(-ac*eta/RTF))

## Interface conditions ##
ise_a = ise(ua_0,u,OCP_a(ca_0/ca_max),io_a(ca_0,c,ca_max))
ise_c = ise(uc_0,u,OCP_c(cc_0/cc_max),io_c(cc_0,c,cc_max))

## Variational formulation ##
n  = FacetNormal(mesh)
F = 0
F += (c-c_k)/dt*d*dx
F += -inner(Ne(u,c), grad(d))*dx
F += -dot(ise_a/Fday*n,d*n)*ds(1)
F += -dot(ise_c/Fday*n,d*n)*ds(2)
F += -inner(je(u,c), grad(v))*dx
F += -dot(ise_a*n,v*n)*ds(1)
F += -dot(ise_c*n,v*n)*ds(2)
F = action(F, U_n)
J = derivative(F, U_n, U)

## pFibs block problem ##
problem = BlockProblem(J,F,U_n,bcs=bcs)

## Built-in fieldsplit for custom schur ##
problem.field('u',0,solver={
    'ksp_type': 'preonly',
    'pc_type': 'hypre',
    'pc_hypre_boomeramg_strong_threshold': 0.75,
    'pc_hypre_boomeramg_agg_nl': 2
})
problem.field('c',1,solver={
    'ksp_type': 'preonly',
    'pc_type': 'hypre',
    'pc_hypre_boomeramg_strong_threshold': 0.75,
    'pc_hypre_boomeramg_agg_nl': 2
})
problem.split('s1',['u','c'],solver={
    'ksp_monitor_true_residual': True
})

## pFibs block solver #
solver = NonlinearBlockSolver(problem)

## Nonlinear solver parameters ##
prm = solver.newton_solver.parameters
prm['absolute_tolerance'] = 1E-17
prm['relative_tolerance'] = 1E-7
prm['maximum_iterations'] = 50
prm['error_on_nonconvergence'] = False
prm['relaxation_parameter'] = 1.0

## Solve ##
startTime = time.time()
iters, converged = solver.solve()
SolveTime = time.time() - startTime

## Show time ##
if rank == 0:
    print("Time: %1.3e seconds" % SolveTime)

