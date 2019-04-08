## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
from numpy import array, where, zeros
from petsc4py import PETSc
from mpi4py import MPI
import copy

## Base class for Python PC ##
class PythonPC(object):
    def __init__(self):

        self.initialized = False
        super(PythonPC, self).__init__()

    ## Setup, do not override ##
    def setUp(self, pc):
        if self.initialized:
            self.update(pc)
        else:
            ## Check for pc_type ##
            if pc.getType() != "python":
                raise ValueError("Expecting PC type python")

            ## Extract the application context ##
            self.ctx = pc.getDM().getAppCtx()
            
            ## Extract the Index Set ##
            self.vbp = self.ctx['problem']
            self.dofs,_ = self.vbp.extract_dofs(self.ctx['field_name'])
            self.isset = PETSc.IS().createGeneral(list(self.dofs))

            ## Determine whether to update the pc or not
            if 'update' in self.ctx:
                self.update_pc = self.ctx['update']
            else:
                self.update_pc = False
		    
            ## Extract options prefix ##
            self.options_prefix = self.ctx['options_prefix'] + 'PythonPC_'

            ## Process PETScOptions ##
            if 'solver' in self.ctx:
                if not isinstance(self.ctx['solver'],dict):
                    raise TypeError('solver must be of type dict')
                self.setPetscOptions(self.options_prefix,self.ctx['solver'])
            
            ## Create KSP object
            self.initialize(pc)
            self.initialized = True
    
    ## Set PETSc options, do not override ##
    def setPetscOptions(self,prefix,options):
        for key in options:
            if type(options[key]) is bool:
                if options[key] is True:
                    df.PETScOptions.set(prefix+key)
            elif options[key] is not None:
                df.PETScOptions.set(prefix+key,options[key])

    ## Extract BC dofs, do not override ##
    def extractBCs(self,bcs):
        if isinstance(bcs,list):
            bc_dofs = []
            bc_value = []
            for bc in bcs:
                sub_bc_dofs, sub_bc_value = self.extractBCs(bc)
                bc_dofs.extend(sub_bc_dofs)
                bc_value.extend(sub_bc_value)
        else:
            # Initialize matrices ##
            bc_dofs = list(bcs.get_boundary_values().keys())
            bc_value = list(bcs.get_boundary_values().values())
            block_dofs = PETSc.IS().createGeneral(bcs.function_space().dofmap().dofs())
            loc2globe = bcs.function_space().dofmap().local_to_global_index
            
            ## Find the indexes of the local boundary dofs ##
            for i in range(len(bc_dofs)):
                dof = bc_dofs[i]
                dof = loc2globe(dof)
                val = where(array(block_dofs) == dof)[0]
                if len(val) == 0:
                    bc_dofs[i] = False
                else:
                    bc_dofs[i] = val[0]
        return bc_dofs, bc_value

    ## Apply BCS to a Matrix, do not override ##
    def applyBCs(self,A,bcs):
        if isinstance(bcs,list):
            for bc in bcs:
                bc.apply(A)
        else:
            bcs.apply(A)

    ## Can override ##
    def initialize(self, pc):
        
        ## Assemble aP ##
        self.P_mat = df.PETScMatrix()
        if 'aP' not in self.ctx:
            raise ValueError("Must provide aP form to ctx")
        else:
            self.aP = self.ctx['aP']
            df.assemble(self.aP, tensor=self.P_mat)

        ## Optionally apply BCs ##
        if 'bcs_aP' in self.ctx:
            self.applyBCs(self.P_mat,self.ctx['bcs_aP'])
        
        ## Extract submatrix ##
        self.P_submat = self.P_mat.mat().createSubMatrix(self.isset,self.isset)
        
        ## Create KSP solver ##
        self.ksp = PETSc.KSP().create(comm=pc.comm)
        self.ksp.setType(PETSc.KSP.Type.PREONLY)
        self.ksp.incrementTabLevel(1, parent=pc)
        self.ksp.setOperators(self.P_submat)
        self.ksp.setOptionsPrefix(self.options_prefix)
        self.ksp.setFromOptions()
        self.ksp.setUp()
   
    ## Can override ##
    def update(self, pc):
        df.assemble(self.aP, tensor=self.P_mat)
        
        ## Optionally apply BCs ##
        if 'bcs_aP' in self.ctx:
            self.applyBCs(self.P_mat,self.ctx['bcs_aP'])

        ## Update submatrix ##
        self.P_submat = self.P_mat.mat().createSubMatrix(self.isset,self.isset,self.P_submat)
        
    ## Can override ##
    def apply(self, pc, x, y):
        self.ksp.solve(x,y)

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
        
#class UvahLiuWu():
#    def __init__(self,H_p,K_p,rho_p):
#        ## This preconditioner will preform:                ##
#        ##                                                  ##
#        ##      y = (1/(2rho)(H+rhoI) (K+rhoI))^(-1) x      ##
#        ##   y = (K+rhoI)^(-1) ((1/(2rho))(H+rhoI))^(-1) x  ##
#
#        ## Store the need operators ##
#        self.H_p = H_p
#        self.K_p = K_p
#        self.rho_p = rho_p
#
#    def build(self):
#        ## Start the timer ##
#        timer = Timer("pFibs: Build Preconditioner")
#
#        ## Assemble Preconditioner ##
#        H_p_mat = PETScMatrix()
#        K_p_mat = PETScMatrix()
#        assemble(self.H_p, tensor=H_p_mat)
#        assemble(self.K_p, tensor=K_p_mat)
#
#        ## Shift by rho ##
#        H_p_mat.mat().shift(self.rho_p)
#        H_p_mat.mat().scale(1.0/(2.0*self.rho_p))
#        K_p_mat.mat().shift(self.rho_p)
#
#        ## Build the solver for the M_p operator ##
#        self.H_p_ksp = PETSc.KSP().create()
#        # self.H_p_ksp.setType(PETSc.KSP.Type.CG)
#        # self.H_p_ksp.pc.setType(PETSc.PC.Type.CHOLESKY)
#        self.H_p_ksp.setType(PETSc.KSP.Type.PREONLY)
#        self.H_p_ksp.pc.setType(PETSc.PC.Type.LU)
#        self.H_p_ksp.setOperators(H_p_mat.mat())
#
#        ## Build the solver for the M_p operator ##
#        self.K_p_ksp = PETSc.KSP().create()
#        # self.K_p_ksp.setType(PETSc.KSP.Type.GMRES)
#        # self.K_p_ksp.pc.setType(PETSc.PC.Type.ILU)
#        self.K_p_ksp.setType(PETSc.KSP.Type.PREONLY)
#        self.K_p_ksp.pc.setType(PETSc.PC.Type.LU)
#        self.K_p_ksp.setOperators(K_p_mat.mat())
#
#        ## Stop the timer ##
#        timer.stop()
#
#    def apply(self, pc, x, y):
#        ## Start the timer ##
#        timer = Timer("pFibs: Apply Preconditioner")
#
#        ## Perform: y = ((1/(2rho))(H+rhoI))^(-1) x ##
#        self.H_p_ksp.solve(x, y)
#
#        ## Copy the result back into x ##
#        x = copy.deepcopy(y)
#
#        ## Perform: y = (K+rhoI)^(-1) x ##
#        self.K_p_ksp.solve(x, y) 
#
#        ## Stop the timer ##
#        timer.stop()
#
#class Elman():
#    def __init__(self,BBT_p,BFBT_p):
#        ## This preconditioner will preform:                ##
#        ##                                                  ##
#        ##   y = (BBT)^(-1) (BFBT)^(-1) x  ##
#
#        ## Store the need operators ##
#        self.BBT_p = BBT_p
#        self.BFBT_p = BFBT_p
#
#    def build(self,is_block):
#        ## Start the timer ##
#        timer = Timer("pFibs: Build Preconditioner")
#
#        ## Store the DOFS related to the block this preconditioner acts on. ##
#        self.block_dofs = is_block
#
#        ## Assemble Preconditioner ##
#        BBT_p_mat = PETScMatrix()
#        BFBT_p_mat = PETScMatrix()
#        assemble(self.BBT_p, tensor=BBT_p_mat)
#        assemble(self.BFBT_p, tensor=BFBT_p_mat)
#
#        ## Get the sub matrix corresponding to the  ##
#        BBT_p_mat = BBT_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
#        self.BFBT_p_mat = BFBT_p_mat.mat().createSubMatrix(self.block_dofs,self.block_dofs)
#
#        ## Build the solver for the M_p operator ##
#        self.BBT_p_ksp = PETSc.KSP().create()
#        self.BBT_p_ksp.setType(PETSc.KSP.Type.PREONLY)
#        self.BBT_p_ksp.pc.setType(PETSc.PC.Type.LU)
#        self.BBT_p_ksp.setOperators(BBT_p_mat)
#
#        ## Stop the timer ##
#        timer.stop()
#
#    def apply(self, pc, x, y):
#        ## Start the timer ##
#        timer = Timer("pFibs: Apply Preconditioner")
#
#        ## Perform: y = BBT^1 x ##
#        self.BBT_p_ksp.solve(x, y)
#
#        ## Apply BFBT_p: x = BFBT y
#        self.BFBT_p_mat.mult(x, y)
#
#        ## Perform: y = BBT^1 x ##
#        self.BBT_p_ksp.solve(x, y)
#
#        ## Stop the timer ##
#        timer.stop()
