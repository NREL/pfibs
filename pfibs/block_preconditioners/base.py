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

