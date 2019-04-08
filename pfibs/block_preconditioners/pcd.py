## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
from pfibs.block_preconditioners.base import PythonPC
from numpy import array, where, zeros
from petsc4py import PETSc
import copy

## Pressure convection diffusion solver     ##
## - Overloads the PythonPC class           ##
## - This preconditioner will perform:      ##
##                                          ##
##    y = -M_p^{-1} (I + K_p A_p^{-1}) x    ##
class PCDPC(PythonPC):
    def __init__(self):
        super(PCDPC,self).__init__()

    ## Build necessary matrices and solvers ##
    def initialize(self, pc):
        if 'vp_spaces' not in self.ctx:
            raise ValueError("Must provide vp_spaces (velocity and pressure subspaces) to ctx")
        else:    
            self.vp_spaces = self.ctx['vp_spaces']
            if not isinstance(self.vp_spaces,list):
                raise TypeError('vp_spaces must be of type list()')
        if 'nu' not in self.ctx:
            raise ValueError('Must provide nu (viscosity) to ctx')
        else:
            self.nu = self.ctx['nu']
        self.V = self.vbp.V
        p = df.TrialFunction(self.V)
        q = df.TestFunction(self.V)
        p = df.split(p)[self.vp_spaces[1]]
        q = df.split(q)[self.vp_spaces[1]]
        u = df.split(self.vbp.u)[self.vp_spaces[0]]

        ## Mass term ##
        self.mP = df.Constant(1.0/self.nu)*p*q*df.dx
        
        ## Advection term ##
        self.aP = df.Constant(1.0/self.nu)*df.dot(df.grad(p), u)*q*df.dx

        ## Stiffness term ##
        self.kP = df.inner(df.grad(p), df.grad(q))*df.dx

        ## Create PETSc Matrices ##
        self.M_p = df.PETScMatrix()
        self.A_p = df.PETScMatrix()
        self.K_p = df.PETScMatrix()
        df.assemble(self.mP, tensor=self.M_p)
        df.assemble(self.aP, tensor=self.A_p)
        df.assemble(self.kP, tensor=self.K_p)
        
        ## Optionally apply BCs ##
        if 'bcs_kP' in self.ctx:
            self.applyBCs(self.K_p,self.ctx['bcs_kP'])
            self.bc_dofs, self.bc_value = self.extractBCs(self.ctx['bcs_kP'])
        
        ## Extract sub matrices ##
        self.M_submat = self.M_p.mat().createSubMatrix(self.isset,self.isset)
        self.A_submat = self.A_p.mat().createSubMatrix(self.isset,self.isset)
        self.K_submat = self.K_p.mat().createSubMatrix(self.isset,self.isset)

        ## KSP solver for mass matrix ##
        self.M_ksp = PETSc.KSP().create(comm=pc.comm)
        self.M_ksp.setType(PETSc.KSP.Type.GMRES)        # Default solver, can change 
        self.M_ksp.pc.setType(PETSc.PC.Type.BJACOBI)    # Default solver, can change
        self.M_ksp.incrementTabLevel(1, parent=pc)
        self.M_ksp.setOperators(self.M_submat)
        self.M_ksp.setOptionsPrefix(self.options_prefix+'mP_')
        self.M_ksp.setFromOptions()
        self.M_ksp.setUp()

        ## KSP solver for stiffness matrix ##
        self.K_ksp = PETSc.KSP().create(comm=pc.comm)
        self.K_ksp.setType(PETSc.KSP.Type.GMRES)        # Default solver, can change 
        self.K_ksp.pc.setType(PETSc.PC.Type.HYPRE)      # Default solver, can change
        self.K_ksp.incrementTabLevel(1, parent=pc)
        self.K_ksp.setOperators(self.K_submat)
        self.K_ksp.setOptionsPrefix(self.options_prefix+'aP_')
        self.K_ksp.setFromOptions()
        self.K_ksp.setUp()

    ## Reassemble non-linear advection term ##
    def update(self,pc):
        
        df.assemble(self.aP, tensor=self.A_p)
        self.A_submat = self.A_p.mat().createSubMatrix(self.isset,self.isset,self.A_submat)
    
    ## Apply PCD preconditioner to pressure schur complement ##
    def apply(self, pc, x, y):
        ## Start the timer ##
        timer = df.Timer("pFibs PythonPC: Apply Preconditioner")

        ## Get working vectors ##
        z = x.duplicate()
        x.copy(result=z)

        ## Apply the boundary conditions to the RHS ##
        z[self.bc_dofs] = self.bc_value

        ## Perform: y = K_submat^{-1} z ##
        self.K_ksp.solve(z, y) # 

        ## Apply A_submat: z = A_submat y
        self.A_submat.mult(y, z)

        ## Add in x: z = z + x ##
        z.axpy(1.0, x)

        ## Perform: y = M_submat^{-1} z ##
        self.M_ksp.solve(z, y)
        
        ## Negate ##
        y.scale(-1.0) 

        ## Stop the timer ##
        timer.stop()
