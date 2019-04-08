## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
import numpy as np
from petsc4py import PETSc

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
        self.aP = kwargs.get("aP",None)
        self.bcs = kwargs.get("bcs",[])
        self.annotate = kwargs.get("annotate",False)
        self.adjoint = kwargs.get("adjoint",False)
        self.ident_zeros = kwargs.get("ident_zeros",False)
        
        ## Extract the Function Space ##
        self.V  = self.u.function_space()
        self.dofs = np.array(self.u.function_space().dofmap().dofs())
        self.goffset = np.min(self.dofs)

        ## Initialize field split information ##
        self.block_field = {}           # Dictionary of all the fields
        self.field_size = {}            # Dictionary of field sizes
        self.block_split = {}           # Dictionary of all the splits
        self.num_fields = 0             # Total number of fields
        self.finalize_field = False     # Flag to indicate all fields added
        self.split_0 = ""               # Name of the outer most split if using nested/recursion

        ## Test if dolfin adjoint is required but not installed ##
        if self.adjoint == True and dolfin_adjoint_found == False:
            raise RuntimeError("Dolfin-adjoint is not installed")
    
    ## Add a field to the block problem ##
    def field(self, *args, **kwargs):
        
        ## Check if splits already defined ##
        if self.finalize_field:
            raise RuntimeError("Cannot add anymore fields after split has been called")

        ## Required input ##
        field_name = args[0]
        field_indx = args[1]

        ## Optional solver parameters ##
        solver_params = kwargs.get("solver",{})

        ## Check types ##
        if not isinstance(field_name, str):
            raise TypeError("Field name must be of type str")
        if not isinstance(solver_params, dict):
            raise TypeError("Solver parameters must be of type dict")
        
        ## Add to dictionary ##
        self.block_field.update({field_name:[self.num_fields,field_indx,solver_params]})
        self.num_fields += 1
 
    ## Extract dofs ##
    def extract_dofs(self,key):
        
        ## If multiple subspaces belong to this field ##
        if isinstance(self.block_field[key][1],list):
            dofs = np.array([])
            for space in self.block_field[key][1]:
        
                ## Case 1: subspace of FunctionSpace ##
                if isinstance(space,int):
                    dofs = np.append(dofs,self.V.sub(space).dofmap().dofs())
               
                ## Case 2: subspace of subspace of FunctionSpace
                elif isinstance(space,list):
                    if len(space) != 2:
                        raise ValueError("Argument length of vector function subspace can only be 2")
                    dofs = np.append(dofs,self.V.sub(space[0]).sub(space[1]).dofmap().dofs())
                else:
                    raise TypeError("Input length must either be an int or a list of ints")
                
        ## Otherwise only one subspace belonging to this field ##
        else:
            dofs = np.array(self.V.sub(self.block_field[key][1]).dofmap().dofs())
        
        ## Get size of array ##
        ndof = dofs.size
        return (dofs, ndof)

    ## Set up the fields ##
    def setup_fields(self):
        
        ## Default if empty ##
        if not self.block_field:
            for i in range(self.V.num_sub_spaces()):
                self.block_field.update({i:[i,i,{}]})
            self.num_fields = len(self.block_field)
        if self.num_fields == 0:
            self.num_fields += 1
        ## Create PetscSection ##
        self.section = PETSc.Section().create()
        self.section.setNumFields(self.num_fields)
        self.section.setChart(0,len(self.V.dofmap().dofs()))

        ## Iterate through all the block fields ##
        for key in self.block_field:
            self.section.setFieldName(self.block_field[key][0],str(key))
            
            ## Extract dofs ##
            (dofs, ndof) = self.extract_dofs(key)
            
            ## Record dof count for each field ##
            self.field_size.update({self.block_field[key][0]:ndof})
            
            ## Assign dof to PetscSection ##
            for i in np.nditer(dofs):
                self.section.setDof(i-self.goffset,1)
                self.section.setFieldDof(i-self.goffset,self.block_field[key][0],1)

        ## Create DM and assign PetscSection ##
        self.section.setUp()
        self.dm = PETSc.DMShell().create()
        self.dm.setDefaultSection(self.section)
        self.dm.setUp()
        
        ## Prevent any further modification to block_field ##
        self.finalize_field = True

    ## Add a split to the block problem ##
    def split(self, *args, **kwargs):

        ## Setup fields ##
        if not self.finalize_field:
            self.setup_fields()

        ## Required input ##
        split_name = args[0]
        split_fields = args[1]
        
        ## Optional solver parameters ##
        solver_params = kwargs.get("solver",{})

        ## Check types ##
        if not isinstance(split_name, str):
            raise TypeError("Field name must be of type str")
        if not isinstance(solver_params, dict):
            raise TypeError("Solver parameters must be of type dict")
        if not isinstance(split_fields, list):
            raise TypeError("Split fields must be of type list")
        elif len(split_fields) < 2:
            raise ValueError("Number of fields in split fields must be 2 or greater")

        ## Check whether split fields exist ##
        for i in split_fields:
            if not isinstance(i,str):
                raise TypeError("Field/split must be of type str")
            if not i in self.block_field and not i in self.block_split:
                raise ValueError("Field/split '%s' not defined" %(i))
        
        ## Add to dictionary ##
        self.block_split.update({split_name:[split_fields,solver_params]})

        ## Update/override as the first split ##
        self._first_split(split_name)

    ## Define the first split ##
    def _first_split(self, split_name):
        if not split_name in self.block_split:
            raise ValueError("First split '%s' not defined" %(split_name))
        else:
            self.split_0 = split_name
    
