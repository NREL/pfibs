## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
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
    def add_field(self, *args, **kwargs):
        
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
    def extractDofs(self,key):
        
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

    ## Return index numbering for fieldsplits ##
    def subIS(self,sub_field_array,full_field_array):
        num_sub_fields = len(sub_field_array)
        sub_field_indx = np.zeros((num_sub_fields,),dtype=np.int32)
        ISarray = np.array([],dtype=np.int32)
        ISsplit = np.zeros(num_sub_fields-1,dtype=np.int32)
        total_field_indx = 0

        ## Allocate the numpy arrays ##
        for i in range(num_sub_fields):
            dof_sum = 0
            if isinstance(sub_field_array[i][1],int):
                dof_sum = self.field_size[sub_field_array[i][1]]
            else:
                for field in sub_field_array[i][1]:
                    dof_sum += self.field_size[field]
            ISarray = np.append(ISarray,np.zeros(dof_sum,dtype=np.int32))
            if i == 0:
                ISsplit[0] = dof_sum
            elif i < num_sub_fields - 1:
                ISsplit[i] = dof_sum + ISsplit[i-1]
        ISarray = np.split(ISarray,ISsplit)

        ## Iterate through Section Chart ##
        pstart,pend = self.section.getChart()
        dofChart = np.arange(pstart,pend)
        for i in np.nditer(dofChart):
            increment = 0
            for field in full_field_array: 
                ## Check if this DoF is associated with the given field ##
                numDof = self.section.getFieldDof(i,field)
                
                ## If it is, find which sub field array it belongs to ##
                if numDof > 0:
                    #if rank == 1:
                    found_field = False
                    for j in range(num_sub_fields):
                        if isinstance(sub_field_array[j][1],int):
                            if field == sub_field_array[j][1]:
                                found_field = True
                        elif field in sub_field_array[j][1]:
                            found_field = True
                        if found_field:
                            ISarray[j][sub_field_indx[j]] = total_field_indx #+ self.goffset
                            sub_field_indx[j] += 1
                            total_field_indx += 1
                            break
                    if not found_field:
                        raise ValueError("field ID %d not found in fieldsplit array" %field)
                    else:
                        break
        
        ## Global offsets ##
        goffset = 0
        if size > 1:
            if rank == 0:
                comm.send(total_field_indx, dest=1)
            elif rank == size - 1:
                goffset = comm.recv(source=rank-1)
            else:
                goffset = comm.recv(source=rank-1)
                comm.send(total_field_indx+goffset,dest=rank+1)
        
        # Create PETSc IS #
        Agg_IS = []
        for i in range(num_sub_fields):
            ISset = np.add(ISarray[i],goffset)
            Agg_IS.append((sub_field_array[i][0],PETSc.IS().createGeneral([ISset])))
        return Agg_IS
       
    ## Set up the fields ##
    def setUpFields(self):
        
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
            (dofs, ndof) = self.extractDofs(key)
            
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
    def add_split(self, *args, **kwargs):

        ## Setup fields ##
        if not self.finalize_field:
            self.setUpFields()

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
        self.first_split(split_name)

    ## Define the first split ##
    def first_split(self, split_name):
        if not split_name in self.block_split:
            raise ValueError("First split '%s' not defined" %(split_name))
        else:
            self.split_0 = split_name
    
