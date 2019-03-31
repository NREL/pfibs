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

## Check if Dolfin Adjoint is installed ##
try:
    import dolfin_adjoint as dfa
    dolfin_adjoint_found = True
except ImportError:
    dolfin_adjoint_found = False

class CustomKrylovSolver(df.PETScKrylovSolver):
    def __init__(self, vbp, options_prefix="",solver={},ctx={}):
        super(CustomKrylovSolver,self).__init__()
        
        ## Initialize field information if it hasn't been done already ##
        self.finalize_field = vbp.finalize_field
        if not self.finalize_field:
            vbp.setup_fields()
        self.field_size = vbp.field_size
        self.num_fields = vbp.num_fields
        self.block_field = vbp.block_field

        ## Copy over split information ##
        self.block_split = vbp.block_split
        self.split_0 = vbp.split_0  
        
        ## Check options prefix type ##
        self.options_prefix = options_prefix
        if not isinstance(self.options_prefix,str):
            raise TypeError("Options prefix must be of type str")
        
        ## Attach options prefix if necessary ##
        if self.options_prefix != "":
            if self.options_prefix[-1] != "_":
                self.options_prefix = self.options_prefix + "_"
            self.ksp().setOptionsPrefix(self.options_prefix)
        
        ## Obtain DM and block problem ##
        self.dm = vbp.dm
        self.section = vbp.section
        self.vbp = vbp
        
        ## Check solver parameters ##
        if not isinstance(solver,dict):
            raise TypeError("Solver parameters must be of type dict()")
        self.solver = solver
        
        ## Check application context ##
        if not isinstance(ctx,dict):
            raise TypeError("Solver context must be of type dict()")
        self.ctx = ctx
        
        ## Attach DM and application context ##
        self.ksp().setDM(self.dm)
        self.ksp().setDMActive(False)
        self.ksp().setAppCtx(self.ctx)
        
    def init_solver_options(self):
 
        ## Start the timer ##
        timer = df.Timer("pFibs: Setup Solver Options")

        ## Set fieldsplit if add_split() not utilized ##
        if self.split_0 == "" and self.num_fields > 1:
            self.ksp().pc.setType(PETSc.PC.Type.FIELDSPLIT)

        ## Setup the PC ##
        self.ksp().setUp()

        ## Manually construct fieldsplits ##
        if self.split_0 != "" and not self.solver:
            self._set_fieldsplit(self.options_prefix,self.split_0,self.ksp(),True)
        
        ## Setup all solver and fieldsplit options via solver dict ##
        ## NOTE: this will override all field/split parameters     ##
        elif self.solver:
            self._set_petsc_options(self.options_prefix,self.block_field[sub_field_array[i][0]][2])

        ## Set PETSc commandline options ##
        self.ksp().setFromOptions()

        ## Stop the timer ##
        timer.stop()
    
    ## Create fieldsplit ##
    def _set_fieldsplit(self, prefix, split_name, subKSP, setpc=True):
        
        ## Obtain block information for this split ##
        block_name = self.block_split[split_name][0]    # Name of the split
        block_solvers = self.block_split[split_name][1] # Solver parameters associated with the split
        n = len(block_name)                             # Number of blocks this split creates
        field_array = []                                # Flattened list of all fields 
        sub_field_array = []                            # List of all fields or subspits per split 

        ## Step 1: find all fields and splits   
        for i in range(n):

            ## Case 1: split contains field ##
            if block_name[i] in self.block_field:
                
                ## Update the lists ##
                sub_field_array.append((str(block_name[i]),[self.block_field[block_name[i]][0]]))
                field_array.append(self.block_field[block_name[i]][0])
                
            ## Case 2: split contains another split ##
            elif block_name[i] in self.block_split:

                ## Flatten list of fields in this split ##
                nested_field_array = self._set_fieldsplit("",block_name[i],subKSP,False)
                
                ## Update the lists ##
                if isinstance(nested_field_array,list):
                    field_array.extend(nested_field_array)
                else:
                    field_array.append(nested_field_array)
                sub_field_array.append((str(block_name[i]),nested_field_array))
            else:
                raise ValueError("Block '%s' not found in the field/split dicts" %(block_name[i]))
        
        ## Optional step 2: create PCFieldSplit ##
        if setpc:
            
            ## Set the fieldsplit ##
            subKSP.pc.setType(PETSc.PC.Type.FIELDSPLIT)

            ## Is it a schur complement ##
            use_schur = self.block_split[split_name][1].get("pc_fieldsplit_type",False)
            if use_schur == 'schur':
                subKSP.pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                       
            ## Assign field index sets to the splits ##
            agg_IS = self._extract_IS(sub_field_array,field_array)
            subKSP.pc.setFieldSplitIS(*agg_IS)
            
            ## Set solver parameters for split ##
            self._set_petsc_options(prefix,block_solvers)
            
            ## Set solver parameters for individual fields if necessary ##
            for i in range(n):
                if sub_field_array[i][0] in self.block_field:
                    # Assign prefix based on field name ##
                    new_prefix = prefix+"fieldsplit_"+sub_field_array[i][0]+"_"
                    self._set_petsc_options(new_prefix,self.block_field[sub_field_array[i][0]][2])
            
            ## Setup the PC and solver options ##
            subKSP.setFromOptions()
            subKSP.pc.setUp()

            ## Split into more sub KSPs ##
            subSubKSP = subKSP.pc.getFieldSplitSubKSP()
            numSubKSP = len(subSubKSP)

            ## Make sure length of subsubKSP samme as n ##
            if numSubKSP != n:
                raise ValueError("Number of sub KSPs = %d but should actually be %d" % (numSubKSP,n))
            
            ## Iterate through all sub KSPs ##
            for i in range(n):

                ## Set application context ##
                subSubKSP[i].setAppCtx(self.ctx)

                ## Recursive fieldsplit if necessary ##
                if sub_field_array[i][0] in self.block_split:
                    self._set_fieldsplit(prefix+"fieldsplit_"+sub_field_array[i][0]+"_",
                            sub_field_array[i][0],subSubKSP[i],True)
        
        ## Return list of all fields in this split ##
        return field_array

    ## Extract index sets for fieldsplit ##
    def _extract_IS(self,sub_field_array,full_field_array):
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
            for field in full_field_array:
                ## Check if this DoF is associated with the given field ##
                numDof = self.section.getFieldDof(i,field)

                ## If it is, find which sub field array it belongs to ##
                if numDof > 0:
                    found_field = False
                    for j in range(num_sub_fields):
                        if isinstance(sub_field_array[j][1],int):
                            if field == sub_field_array[j][1]:
                                found_field = True
                        elif field in sub_field_array[j][1]:
                            found_field = True
                        if found_field:
                            ISarray[j][sub_field_indx[j]] = total_field_indx
                            sub_field_indx[j] += 1
                            total_field_indx += 1
                            break
                    if not found_field:
                        raise ValueError("field ID %d not found in fieldsplit array" %field)
                    else:
                        break

        ## Global offsets communicated via MPI_Scan ##
        goffset = comm.scan(total_field_indx) - total_field_indx

        # Create PETSc IS #
        agg_IS = []
        for i in range(num_sub_fields):
            ISset = np.add(ISarray[i],goffset)
            agg_IS.append((sub_field_array[i][0],PETSc.IS().createGeneral([ISset])))
        return agg_IS

    ## Set PETScOptions parameters ##
    def _set_petsc_options(self,prefix,solver_params):
        for key in solver_params:
            if type(solver_params[key]) is bool:
                if solver_params[key] is True:
                    df.PETScOptions.set(prefix+key)
            elif solver_params[key] is not None:
                df.PETScOptions.set(prefix+key,solver_params[key])
        
### This stuff is not needed anymore. Keeping it here for future reference ###
#
#    def set_pc_type(self,*args):
#        ## This method takes either a string or an int and string. Providing just a string      ##
#        ## attempts to set the pc of the full system. Providing a int and a string attempts to  ##
#        ## set the preconditioner of a block (indicated by the int). Finally, if list of ints   ##
#        ## is provided, all blocks in that list will have the pc set to the indicated type.     ##
#
#        ## Check if only string is provided ##
#        if len(args) == 1:
#            if isinstance(args[0], str):
#
#                ## Check if string is a valid pc type ##
#                if hasattr(PETSc.PC.Type,args[0].upper()):
#                    self.ksp().pc.setType(args[0])
#                else:
#                    raise RuntimeError('Not a valid preconditioner')
#
#            ## A class can also be provided for a custom preconditioner ##
#            else:
#                self.ksp().pc.setType(PETSc.PC.Type.PYTHON)
#                args[0].build()
#                self.ksp().pc.setPythonContext(args[0])
#
#        ## Check if int/list and string/class is provided
#        elif len(args) == 2 and (isinstance(args[0], int), isinstance(args[0], str) or isinstance(args[0], list)):
#            if not isinstance(args[0], list):
#                blocks = [args[0]]
#            else:
#                blocks = args[0]
#            for block in blocks:
#                block = self.block_dict[block]
#                if isinstance(args[1], str):
#                    ## Check if string is a valid pc type ##
#                    if hasattr(PETSc.PC.Type,args[1].upper()):
#                        self.subksp[block].pc.setType(args[1])
#                    else:
#                        raise RuntimeError('Not a valid preconditioner')
#                
#                ## A class can also be provided for a custom preconditioner ##
#                else:
#                    self.subksp[block].pc.setType(PETSc.PC.Type.PYTHON)
#                    args[1].build(self.block_dofs[block])
#                    self.subksp[block].pc.setPythonContext(args[1])
#        
#        ## If all fails, report the types ##
#        else:
#            raise TypeError('To set overall preconditioner: arg1: str (pc), \
#                \nTo set a preconditioner on a specific block: arg1: int/str (block), arg2: str/class (pc) \
#                \nTo set a preconditioner on a multiple blocks: arg1: list (blocks), arg2: str/class (pc)')
#
#    def set_ksp_type(self,*args):
#        ## This method takes either a string or an int and string. Providing just a string      ##
#        ## attempts to set the ksp_type of the full system. Providing a int and a string        ##
#        ## attempts to set the ksp_type of a block (indicated by the int). Finally, if list of  ##
#        ## ints is provided, all blocks in that list will have the ksp set to the indicated     ##
#        ## type. 
#
#        ## Check if only string is provided ##
#        if len(args) == 1 and isinstance(args[0], str):
#
#            ## Check if string is a valid pc type ##
#            if hasattr(PETSc.KSP.Type,args[0].upper()):
#                self.ksp().setType(args[0])
#            else:
#                raise RuntimeError('Not a valid KSP Type')
#
#        ## Check if int/list and string/class is provided
#        elif len(args) == 2 and (isinstance(args[0], int), isinstance(args[0], str) or isinstance(args[0], list)):
#            if not isinstance(args[0], list):
#                blocks = [args[0]]
#            else:
#                blocks = args[0]
#            for block in blocks:
#                block = self.block_dict[block]
#                if isinstance(args[1], str):
#
#                    ## Check if string is a valid ksp type ##
#                    if hasattr(PETSc.KSP.Type,args[1].upper()):
#                        self.subksp[block].setType(args[1])
#                    else:
#                        raise RuntimeError('Not a valid KSP Type')
#                        
#        ## If all fails, report the types ##        
#        else:
#            raise TypeError('To set overall ksp_type: arg1: str (pc), \
#                \nTo set a ksp_type on a specific block: arg1: str/int (block), arg2: str/class (pc) \
#                \nTo set a ksp_type on a multiple blocks: arg1: list (blocks), arg2: str/class (pc)')
#
#    def set_schur(self,*args):
#        if self.num_blocks != 2:
#            raise RuntimeError("Can only use Schur with 2 blocks")
#
#        ## Reset the ksp to it can be reinitialized as schur ##
#        # self.ksp().reset()
#        self.ksp().pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
#        self.ksp().pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
#
#        ## set schur preconditioner type if specified ##
#        if len(args) >= 1:
#            if hasattr(PETSc.PC.SchurPreType,args[0].upper()):
#                self.ksp().pc.setFieldSplitSchurPreType(getattr(PETSc.PC.SchurPreType, args[0].upper()))
#            else:
#                raise RuntimeError('Not a valid Schur Type')
#
#
