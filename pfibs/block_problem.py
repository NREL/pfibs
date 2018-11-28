## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df

## Optionally import dolfin_adjoint ##
try:
    import dolfin_adjoint as dfa 
    dolfin_adjoint_found = True
except ImportError:
    dolfin_adjoint_found = False

class BlockProblem(object):
    def __init__(self, *args, **kwargs):

        ## Check if correct number of positional arguments ##
        if len(args) != 3:
            raise RuntimeError("Solver takes only three positional augments: a,L,x")

        ## Store Problem Info ##
        self.a  = args[0]
        self.L  = args[1]
        self.u  = args[2]
        self.bcs = kwargs.get("bcs",[])
        self.annotate = kwargs.get("annotate",False)
        self.adjoint = kwargs.get("adjoint",False)
        self.ident_zeros = kwargs.get("ident_zeros",False)

        ## Extract the Function Space ##
        self.V  = self.u.function_space()
        self.dofs  = self.u.function_space().dofmap().dofs()

        ## Setup the block structure ##
        self.setBlockStructure(kwargs.get("block_structure",None))

        ## Dictionary for KSP and PC types ##
        self.schur = None
        self.ksp_type = "gmres"
        self.pc_type = "fieldsplit"
        self.sub_ksp_type = dict.fromkeys(self.block_dict,"preonly")
        self.sub_pc_type = dict.fromkeys(self.block_dict,"ilu")

        ## Test if dolfin adjoint is required but not installed ##
        if self.adjoint == True and dolfin_adjoint_found == False:
            raise RuntimeError("Dolfin-adjoint is not installed")
    
    def setBlockStructure(self,block_structure):
        ## Build default block structure if not provided. By default, we will split according ##
        ## to the number of subspace on the first level of the function space                 ##
        if block_structure is None:
            self.block_structure = []
            for i in range(self.V.num_sub_spaces()):
                self.block_structure += [[i,[i]]]
        else:
            self.block_structure = block_structure

        ## Define the number of blocks ##
        self.num_blocks = len(self.block_structure)

        ## Create a dictionary from the block structure ##
        self.block_dict = dict()
        for i in range(len(self.block_structure)):
            self.block_dict.update( {self.block_structure[i][0]:i} )

    def KSPType(self,ksp_type):
        self.ksp_type = ksp_type

    def PCType(self,pc_type):
        self.pc_type = pc_type

    def SchurType(self,schur_type):
        self.schur=schur_type 

    def SubKSPType(self,block,ksp_type):
        self.sub_ksp_type[block]=ksp_type 

    def SubPCType(self,block,pc_type):
        self.sub_pc_type[block]=pc_type 


