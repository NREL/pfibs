## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
from dolfin import PETScMatrix, PETScVector, assemble, Timer
from numpy import array, where, zeros
from petsc4py import PETSc
from mpi4py import MPI
import copy

class PCD_BRM1(object):
    #def __init__(self):
        ## This preconditioner will preform:        ##
        ##                                          ##
        ##    y = -M_p^{-1} (I + K_p A_p^{-1}) x    ##

    #    self.initialized = False
    #    super(PCD_BRM1, self).__init__()

    ## User is required to implement the following ##
    def build(self):
        pass

    ## PETSc method, either initialize or update the PC ##
    def setUp(self, pc):
        #self.build(pc)
        print("HE1")
        ctx = self.pc.getKSP().getDM().getAppCtx()
        print(ctx)
        self.initialize(pc)
        #print(self.hi)
        #exit()
        #if self.initialized:
        #    self.update(pc)
        #else:
        #    self.build(pc)
        #    self.initialize(pc)
        #    self.intialized = True
    ## 
    def initialize(self, pc):
        timer = Timer("pFibs: Initialize Preconditioner")
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        print(self.test)
        exit()
        A,P = pc.getOperators()
        print(A.getType())
        print("HEY1")
        ctx = pc.getKSP().content
        #ctx = P.getPythonContext()
        print("HEY2")
        #ctx = pc.getPythonContext()
        print(ctx.ctx)
        ## Setup bcs and submatrices ##
        print("HEY3")
        self.bc = ctx["bc"]
        
        self.bc_dofs = list(self.bc.get_boundary_values().keys())
        self.bc_value = list(self.bc.get_boundary_values().values())
        self.block_dofs = PETSc.IS().createGeneral(self.bc.function_space().dofmap().dofs())
        loc2globe = self.bc.function_space().dofmap().local_to_global_index
        
        ## Find the indexes of the local boundary dofs ##
        for i in range(len(self.bc_dofs)):
            dof = self.bc_dofs[i]
            dof = loc2globe(dof)
            val = where(array(self.block_dofs) == dof)[0]
            if len(val) == 0:
                self.bc_dofs[i] = False
            else:
                self.bc_dofs[i] = val[0]
        timer2.stop()
        
        ## Extract PCD operators ##
        self.M_p = ctx["M_p"]
        self.K_p = ctx["K_p"]
        self.A_p = ctx["A_p"]
        M_p_mat = PETScMatrix()
        K_p_mat = PETScMatrix()
        A_p_mat = PETScMatrix()
        assemble(self.M_p, tensor=M_p_mat)
        assemble(self.K_p, tensor=K_p_mat)
        assemble(self.A_p, tensor=A_p_mat)
        self.bc.apply(A_p_mat)
        
        ## Extract submatrices to use ##
        M_p_mat = M_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
        self.K_p_matat = K_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
        A_p_mat = A_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
        
        ## Build the solver for the M_p operator ##
        self.M_p_ksp = PETSc.KSP().create()
        self.M_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.M_p_ksp.pc.setType(PETSc.PC.Type.HYPRE)
        self.M_p_ksp.setOperators(M_p_mat)

        ## Build the solver for the A_p operator ##
        self.A_p_ksp = PETSc.KSP().create()
        self.A_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.A_p_ksp.pc.setType(PETSc.PC.Type.HYPRE)
        self.A_p_ksp.setOperators(A_p_mat)
        
        timer.stop()  

    def update(self, pc):
        ssemble(self.M_p, tensor=M_p_mat)
        assemble(self.K_p, tensor=K_p_mat)
        assemble(self.A_p, tensor=A_p_mat)
        self.bc.apply(A_p_mat)

    def build(self,pc):
        ## Start the timer ##
        timer = Timer("pFibs: Build Preconditioner")

        ## Store the DOFS related to the block this preconditioner acts on. ##
        self.block_dofs = is_block
        loc2globe = self.bc.function_space().dofmap().local_to_global_index

        timer2 = Timer("pFibs: Find Block Boundary DOFS")
        ## Find the indexes of the local boundary dofs ##
        for i in range(len(self.bc_dofs)):
            dof = self.bc_dofs[i]
            dof = loc2globe(dof)
            val = where(array(self.block_dofs) == dof)[0]
            if len(val) == 0:
                self.bc_dofs[i] = False
            else:
                self.bc_dofs[i] = val[0]
        timer2.stop()

        ## Assemble Preconditioner, apply bcs, and extract the relevant submatrices. ##
        M_p_mat = PETScMatrix()
        K_p_mat = PETScMatrix()
        A_p_mat = PETScMatrix()
        assemble(self.M_p, tensor=M_p_mat)
        assemble(self.K_p, tensor=K_p_mat)
        assemble(self.A_p, tensor=A_p_mat)
        self.bc.apply(A_p_mat)
        M_p_mat = M_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
        self.K_p_matat = K_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
        A_p_mat = A_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)

        ## Build the solver for the M_p operator ##
        self.M_p_ksp = PETSc.KSP().create()
        self.M_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.M_p_ksp.pc.setType(PETSc.PC.Type.HYPRE)
        self.M_p_ksp.setOperators(M_p_mat)

        ## Build the solver for the M_p operator ##
        self.A_p_ksp = PETSc.KSP().create()
        self.A_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.A_p_ksp.pc.setType(PETSc.PC.Type.HYPRE)
        self.A_p_ksp.setOperators(A_p_mat)

        ## Stop the timer ##
        timer.stop()

    def apply(self, pc, x, y):
        ## Start the timer ##
        timer = Timer("pFibs: Apply Preconditioner")

        ## I'm not sure what this does but the fenapack guys do it so... ##
        z = x.duplicate()

        ## Make a copy of so we can use x later ##
        x.copy(result=z)

        ## Apply the boundary conditions to the RHS ##
        # self.bc.apply(z)
        z[self.bc_dofs] = self.bc_value

        ## Perform: y = A_p^{-1} z ##
        self.A_p_ksp.solve(z, y) # 

        ## Apply K_p: z = K_p y
        self.K_p_matat.mult(y, z)

        ## Add in x: z = z + x ##
        z.axpy(1.0, x)

        ## Perform: y = M_p^{-1} z ##
        self.M_p_ksp.solve(z, y)
        
        ## Negate ##
        y.scale(-1.0) 

        ## Stop the timer ##
        timer.stop()

class MyPCD(PCD_BRM1):
    def initialize(self,pc):
        print("HI FROM MYPCD")




class Pre_Laplace():
    def __init__(self,A_p):
        ## This preconditioner will preform:        ##
        ##                                          ##
        ##    y = -M_p^{-1} (I + K_p A_p^{-1}) x    ##

        ## Store the need operators ##
        self.A_p = A_p

    def build(self,is_block):
        ## Start the timer ##
        timer = Timer("pFibs: Build Preconditioner")

        ## Store the DOFS related to the block this preconditioner acts on. ##
        self.block_dofs = is_block

        ## Assemble Preconditioner, apply bcs, and extract the relevant submatrices. ##
        A_p_mat = PETScMatrix()
        assemble(self.A_p, tensor=A_p_mat)
        A_p_mat.ident_zeros()
        A_p_mat = A_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)

        ## Build the solver for the M_p operator ##
        self.A_p_ksp = PETSc.KSP().create()
        self.A_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.A_p_ksp.pc.setType(PETSc.PC.Type.HYPRE)
        self.A_p_ksp.setOperators(A_p_mat)

        ## Stop the timer ##
        timer.stop()

    def apply(self, pc, x, y):
        ## Start the timer ##
        timer = Timer("pFibs: Apply Preconditioner")

        ## Perform: y = A_p^{-1} x ##
        self.A_p_ksp.solve(x, y) # 

        ## Stop the timer ##
        timer.stop()


class UvahLiuWu():
    def __init__(self,H_p,K_p,rho_p):
        ## This preconditioner will preform:                ##
        ##                                                  ##
        ##      y = (1/(2rho)(H+rhoI) (K+rhoI))^(-1) x      ##
        ##   y = (K+rhoI)^(-1) ((1/(2rho))(H+rhoI))^(-1) x  ##

        ## Store the need operators ##
        self.H_p = H_p
        self.K_p = K_p
        self.rho_p = rho_p

    def build(self):
        ## Start the timer ##
        timer = Timer("pFibs: Build Preconditioner")

        ## Assemble Preconditioner ##
        H_p_mat = PETScMatrix()
        K_p_mat = PETScMatrix()
        assemble(self.H_p, tensor=H_p_mat)
        assemble(self.K_p, tensor=K_p_mat)

        ## Shift by rho ##
        H_p_mat.mat().shift(self.rho_p)
        H_p_mat.mat().scale(1.0/(2.0*self.rho_p))
        K_p_mat.mat().shift(self.rho_p)

        ## Build the solver for the M_p operator ##
        self.H_p_ksp = PETSc.KSP().create()
        # self.H_p_ksp.setType(PETSc.KSP.Type.CG)
        # self.H_p_ksp.pc.setType(PETSc.PC.Type.CHOLESKY)
        self.H_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.H_p_ksp.pc.setType(PETSc.PC.Type.LU)
        self.H_p_ksp.setOperators(H_p_mat.mat())

        ## Build the solver for the M_p operator ##
        self.K_p_ksp = PETSc.KSP().create()
        # self.K_p_ksp.setType(PETSc.KSP.Type.GMRES)
        # self.K_p_ksp.pc.setType(PETSc.PC.Type.ILU)
        self.K_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.K_p_ksp.pc.setType(PETSc.PC.Type.LU)
        self.K_p_ksp.setOperators(K_p_mat.mat())

        ## Stop the timer ##
        timer.stop()

    def apply(self, pc, x, y):
        ## Start the timer ##
        timer = Timer("pFibs: Apply Preconditioner")

        ## Perform: y = ((1/(2rho))(H+rhoI))^(-1) x ##
        self.H_p_ksp.solve(x, y)

        ## Copy the result back into x ##
        x = copy.deepcopy(y)

        ## Perform: y = (K+rhoI)^(-1) x ##
        self.K_p_ksp.solve(x, y) 

        ## Stop the timer ##
        timer.stop()





class Elman():
    def __init__(self,BBT_p,BFBT_p):
        ## This preconditioner will preform:                ##
        ##                                                  ##
        ##   y = (BBT)^(-1) (BFBT)^(-1) x  ##

        ## Store the need operators ##
        self.BBT_p = BBT_p
        self.BFBT_p = BFBT_p

    def build(self,is_block):
        ## Start the timer ##
        timer = Timer("pFibs: Build Preconditioner")

        ## Store the DOFS related to the block this preconditioner acts on. ##
        self.block_dofs = is_block

        ## Assemble Preconditioner ##
        BBT_p_mat = PETScMatrix()
        BFBT_p_mat = PETScMatrix()
        assemble(self.BBT_p, tensor=BBT_p_mat)
        assemble(self.BFBT_p, tensor=BFBT_p_mat)

        ## Get the sub matrix corresponding to the  ##
        BBT_p_mat = BBT_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
        self.BFBT_p_mat = BFBT_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)

        ## Build the solver for the M_p operator ##
        self.BBT_p_ksp = PETSc.KSP().create()
        self.BBT_p_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.BBT_p_ksp.pc.setType(PETSc.PC.Type.LU)
        self.BBT_p_ksp.setOperators(BBT_p_mat)

        ## Stop the timer ##
        timer.stop()

    def apply(self, pc, x, y):
        ## Start the timer ##
        timer = Timer("pFibs: Apply Preconditioner")

        ## Perform: y = BBT^1 x ##
        self.BBT_p_ksp.solve(x, y)

        ## Apply BFBT_p: x = BFBT y
        self.BFBT_p_mat.mult(x, y)

        ## Perform: y = BBT^1 x ##
        self.BBT_p_ksp.solve(x, y)

        ## Stop the timer ##
        timer.stop()




class CustomPreTemplate():
    def __init__(self,SomeMatrix):
        ## Use this to store the relevant matrices ##
        self.SomeMatrix = SomeMatrix

    def build(self):
        ## Use this to assemble matrices and build individual ksp solvers ##

        ## Start the timer ##
        timer = Timer("pFibs: Build Preconditioner")

        ## Assemble Preconditioner ##
        SomeMatrix_mat = PETScMatrix()
        assemble(self.SomeMatrix, tensor=SomeMatrix_mat)

        ## Build the solver for the M_p operator ##
        self.SomeMatrix_ksp = PETSc.KSP().create()
        self.SomeMatrix_ksp.setType(PETSc.KSP.Type.PREONLY)
        self.SomeMatrix_ksp.pc.setType(PETSc.PC.Type.LU)
        self.SomeMatrix_ksp.setOperators(SomeMatrix_mat.mat())

        ## Stop the timer ##
        timer.stop()

    def apply(self, pc, x, y):
        ## Use this to apply the preconditioner ##

        ## Start the timer ##
        timer = Timer("pFibs: Apply Preconditioner")

        ## Perform: y = (SomeMatrix)^(-1) x ##
        self.SomeMatrix_ksp.solve(x, y)

        ## Stop the timer ##
        timer.stop()
