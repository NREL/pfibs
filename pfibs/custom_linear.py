## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
from petsc4py import PETSc

## Check if Dolfin Adjoint is installed ##
try:
    import dolfin_adjoint as dfa
    dolfin_adjoint_found = True
except ImportError:
    dolfin_adjoint_found = False

class CustomKrylovSolver(df.PETScKrylovSolver):
    def __init__(self, vbp, options_prefix):
        super(CustomKrylovSolver,self).__init__()

        self.block_structure = vbp.block_structure
        self.block_dict = vbp.block_dict
        self.num_blocks = vbp.num_blocks
        self.V = vbp.V
        self.dofs = vbp.dofs

        self.schur = vbp.schur
        self.ksp_type = vbp.ksp_type
        self.pc_type = vbp.pc_type
        self.sub_ksp_type = vbp.sub_ksp_type
        self.sub_pc_type = vbp.sub_pc_type
    
        if options_prefix is not None:
            self.ksp().setOptionsPrefix(options_prefix+"_")

    def init_solver_options(self):
 
        ## Start the timer ##
        timer = df.Timer("pFibs: Setup Solver Options")

        ## Set global KSP ##
        self.set_ksp_type(self.ksp_type)

        ## Set global PC ## 
        self.set_pc_type(self.pc_type)

        ## Deal with fieldsplitting ##
        if self.pc_type is "fieldsplit":

            ## Setup the Fields ##
            self.setup_field_split()

            ## Setup schur if defined ##
            if self.schur is not None:
                self.set_schur(self.schur)
                self.ksp().pc.setUp()

            ## Split the preconditioner into the individual blocks. ##
            self.subksp = self.ksp().pc.getFieldSplitSubKSP()

            # ## Set sub KSP ##
            for block in self.sub_ksp_type:
                self.set_ksp_type(block,self.sub_ksp_type[block])
            
            # ## Set sub PC ##
            for block in self.sub_pc_type:
                self.set_pc_type(block,self.sub_pc_type[block])

            ## Set PETSc commandline options for subksp ##
            for ksp in self.subksp:
                ksp.setFromOptions()

        ## Set PETSc commandline options ##
        self.ksp().setFromOptions()

        ## Stop the timer ##
        timer.stop()

    def setup_field_split(self):
        ## Extract degrees of freedom associated with the desired blocks ##
        self.block_dofs = [()]*self.num_blocks
        self.block_names = [""]*self.num_blocks
        for block in range(self.num_blocks):
            self.block_names[block] = self.block_structure[block][0]
            if isinstance(self.block_structure[block][1],list):
                dofs = []
                for space in self.block_structure[block][1]:
                    if isinstance(space,int):
                        dofs += list(self.V.sub(space).dofmap().dofs())
                    elif isinstance(space,list):
                        if len(space) != 2:
                            raise RuntimeError("Argument length  of vector function subspace can only be 2")
                        dofs += list(self.V.sub(space[0]).sub(space[1]).dofmap().dofs())
                    else:
                        raise RuntimeError("Input length must either be an int or a list of ints")
                dofs.sort()
            else:
                dofs = list(self.V.sub(block).dofmap().dofs())
            self.block_dofs[block] = PETSc.IS().createGeneral(dofs)
            self.ksp().pc.setFieldSplitIS([str(self.block_names[block]), self.block_dofs[block]])


    def set_pc_type(self,*args):
        ## This method takes either a string or an int and string. Providing just a string      ##
        ## attempts to set the pc of the full system. Providing a int and a string attempts to  ##
        ## set the preconditioner of a block (indicated by the int). Finally, if list of ints   ##
        ## is provided, all blocks in that list will have the pc set to the indicated type.     ##

        ## Check if only string is provided ##
        if len(args) == 1:
            if isinstance(args[0], str):

                ## Check if string is a valid pc type ##
                if hasattr(PETSc.PC.Type,args[0].upper()):
                    self.ksp().pc.setType(args[0])
                else:
                    raise RuntimeError('Not a valid preconditioner')

            ## A class can also be provided for a custom preconditioner ##
            else:
                self.ksp().pc.setType(PETSc.PC.Type.PYTHON)
                args[0].build()
                self.ksp().pc.setPythonContext(args[0])

        ## Check if int/list and string/class is provided
        elif len(args) == 2 and (isinstance(args[0], int), isinstance(args[0], str) or isinstance(args[0], list)):
            if not isinstance(args[0], list):
                blocks = [args[0]]
            else:
                blocks = args[0]
            for block in blocks:
                block = self.block_dict[block]
                if isinstance(args[1], str):
                    ## Check if string is a valid pc type ##
                    if hasattr(PETSc.PC.Type,args[1].upper()):
                        self.subksp[block].pc.setType(args[1])
                    else:
                        raise RuntimeError('Not a valid preconditioner')
                
                ## A class can also be provided for a custom preconditioner ##
                else:
                    self.subksp[block].pc.setType(PETSc.PC.Type.PYTHON)
                    args[1].build(self.block_dofs[block])
                    self.subksp[block].pc.setPythonContext(args[1])
        
        ## If all fails, report the types ##
        else:
            raise TypeError('To set overall preconditioner: arg1: str (pc), \
                \nTo set a preconditioner on a specific block: arg1: int/str (block), arg2: str/class (pc) \
                \nTo set a preconditioner on a multiple blocks: arg1: list (blocks), arg2: str/class (pc)')

    def set_ksp_type(self,*args):
        ## This method takes either a string or an int and string. Providing just a string      ##
        ## attempts to set the ksp_type of the full system. Providing a int and a string        ##
        ## attempts to set the ksp_type of a block (indicated by the int). Finally, if list of  ##
        ## ints is provided, all blocks in that list will have the ksp set to the indicated     ##
        ## type. 

        ## Check if only string is provided ##
        if len(args) == 1 and isinstance(args[0], str):

            ## Check if string is a valid pc type ##
            if hasattr(PETSc.KSP.Type,args[0].upper()):
                self.ksp().setType(args[0])
            else:
                raise RuntimeError('Not a valid KSP Type')

        ## Check if int/list and string/class is provided
        elif len(args) == 2 and (isinstance(args[0], int), isinstance(args[0], str) or isinstance(args[0], list)):
            if not isinstance(args[0], list):
                blocks = [args[0]]
            else:
                blocks = args[0]
            for block in blocks:
                block = self.block_dict[block]
                if isinstance(args[1], str):

                    ## Check if string is a valid ksp type ##
                    if hasattr(PETSc.KSP.Type,args[1].upper()):
                        self.subksp[block].setType(args[1])
                    else:
                        raise RuntimeError('Not a valid KSP Type')
                        
        ## If all fails, report the types ##        
        else:
            raise TypeError('To set overall ksp_type: arg1: str (pc), \
                \nTo set a ksp_type on a specific block: arg1: str/int (block), arg2: str/class (pc) \
                \nTo set a ksp_type on a multiple blocks: arg1: list (blocks), arg2: str/class (pc)')

    def set_schur(self,*args):
        if self.num_blocks != 2:
            raise RuntimeError("Can only use Schur with 2 blocks")

        ## Reset the ksp to it can be reinitialized as schur ##
        # self.ksp().reset()
        self.ksp().pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        self.ksp().pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

        ## set schur preconditioner type if specified ##
        if len(args) >= 1:
            if hasattr(PETSc.PC.SchurPreType,args[0].upper()):
                self.ksp().pc.setFieldSplitSchurPreType(getattr(PETSc.PC.SchurPreType, args[0].upper()))
            else:
                raise RuntimeError('Not a valid Schur Type')


