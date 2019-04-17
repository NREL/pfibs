## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from helper import iterate_section
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
        
        self.log_level = vbp.log_level

        if self.log_level >= 1:
            ## Time function execution ##
            timer = df.Timer("pFibs: Init custom Krylov solver")
        
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

        ## Attach block problem to context. Used to extract sub matrices ##
        self.ctx.update({'problem':self.vbp})
        
        ## Attach the overall solver prefix ##
        self.ctx.update({'prefix':self.options_prefix})

        ## Attach DM and application context ##
        self.dm.setAppCtx(self.ctx)
        self.ksp().setDM(self.dm)
        self.ksp().setDMActive(False)
        self.ksp().setAppCtx(self.ctx)
        
        ## Split DM into subDMs ##
        _, _, self.subdms = self.dm.createFieldDecomposition()

        if self.log_level >= 1:
            timer.stop()
        
    def init_solver_options(self):
 
        ## Start the timer ##
        timer = df.Timer("pFibs: Setup Solver Options")

        if self.log_level >= 1:
            timer_fieldsplit = df.Timer("pFibs: Setup Solver Options - set_fieldsplit")
        ## Manually construct fieldsplit if more than one field detected ##
        if not self.solver and self.split_0 != "":
            self._set_fieldsplit(self.options_prefix,self.split_0,self.ksp(),True)
        
        ## Define fieldsplit if no split was called ##
        elif self.split_0 == "" and self.num_fields > 1 and not self.solver:
            self.vbp.split('s',list(self.block_field.keys()))
            self.split_0 = self.vbp.split_0
            self._set_fieldsplit(self.options_prefix,self.split_0,self.ksp(),True)
        
        ## Or setup all solver and fieldsplit options via solver dict ##
        elif self.solver:
            self._set_petsc_options(self.options_prefix,self.solver)
        if self.log_level >= 1:
            timer_fieldsplit.stop()

        if self.log_level >= 1:
            timer1 = df.Timer("pFibs: Setup Solver Options - setup PETSC commandline options")

        ## Set PETSc commandline options ##
        self.ksp().setFromOptions()
        self.ksp().setUp()

        if self.log_level >= 1:
            timer1.stop()

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
                
                if self.log_level >= 2:
                    timer_findfields = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - find fields and splits")
                ## Update the lists ##
                sub_field_array.append((str(block_name[i]),[self.block_field[block_name[i]][0]]))
                field_array.append(self.block_field[block_name[i]][0])
                if self.log_level >= 2:
                    timer_findfields.stop()
                
            ## Case 2: split contains another split ##
            elif block_name[i] in self.block_split:

                ## Flatten list of fields in this split ##
                nested_field_array = self._set_fieldsplit("",block_name[i],subKSP,False)
                
                if self.log_level >= 2:
                    timer_updatelists = df.Timer("pFibs - Setup Solver Options - set_fieldsplit - update lists")
                ## Update the lists ##
                if isinstance(nested_field_array,list):
                    field_array.extend(nested_field_array)
                else:
                    field_array.append(nested_field_array)
                sub_field_array.append((str(block_name[i]),nested_field_array))
                if self.log_level >= 2:
                    timer_updatelists.stop()
            else:
                raise ValueError("Block '%s' not found in the field/split dicts" %(block_name[i]))
        
        ## Optional step 2: create PCFieldSplit ##
        if setpc:
            
            if self.log_level >= 2:
                timer_createPCfieldsplit = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - create PCFieldSplit")
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

                ## Recursive fieldsplit if necessary ##
                if sub_field_array[i][0] in self.block_split:
                    if self.log_level >= 2:
                        timer_createPCfieldsplit.stop()
                    self._set_fieldsplit(prefix+"fieldsplit_"+sub_field_array[i][0]+"_",
                            sub_field_array[i][0],subSubKSP[i],True)
                    if self.log_level >= 2:
                        timer_createPCfieldsplit.start()
                ## Set application context ##
                else:
                    if self.log_level >= 3:
                        timer_setappcont = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - create PCFieldSplit - set application context")
                    subctx = {}
                    subctx.update(self.ctx)
                    subctx.update({'field_name':sub_field_array[i][0]})
                    subctx.update({'options_prefix':prefix+"fieldsplit_"+sub_field_array[i][0]+"_"})
                    subdm = self.subdms[self.block_field[sub_field_array[i][0]][0]]
                    subdm.setAppCtx(subctx)
                    subSubKSP[i].setDM(subdm)
                    subSubKSP[i].setDMActive(False)
                    if self.log_level >= 3:
                        timer_setappcont.stop()
            
            if self.log_level >= 2:
                timer_createPCfieldsplit.stop()
        ## Return list of all fields in this split ##
        return field_array

    ## Extract index sets for fieldsplit ##
    def _extract_IS(self,sub_field_array,full_field_array):
        if self.log_level >= 3:
            timer = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - create PCFieldSplit - extract_IS")
        num_sub_fields = len(sub_field_array)
        sub_field_indx = np.zeros((num_sub_fields,),dtype=np.int32)
        ISarray = np.array([],dtype=np.int32)
        ISsplit = np.zeros(num_sub_fields-1,dtype=np.int32)
        total_field_indx = 0

        if self.log_level >= 4:
            timer_allocNParrays = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - create PCFieldSplit - extract_IS - allocate numpy arrays")
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
        if self.log_level >= 4:
            timer_allocNParrays.stop()

        if self.log_level >= 4:
            timer_iterSecChart = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - create PCFieldSplit - extract_IS - iterate through Section Chart")
        iterate_section(self.section, full_field_array, num_sub_fields, sub_field_array, total_field_indx, ISarray, sub_field_indx)
        ## Iterate through Section Chart ##
#        pstart,pend = self.section.getChart()
#        dofChart = np.arange(pstart,pend)
#        for i in np.nditer(dofChart):
#            for field in full_field_array:
#                ## Check if this DoF is associated with the given field ##
#                numDof = self.section.getFieldDof(i,field)
#
#                ## If it is, find which sub field array it belongs to ##
#                if numDof > 0:
#                    found_field = False
#                    for j in range(num_sub_fields):
#                        if isinstance(sub_field_array[j][1],int):
#                            if field == sub_field_array[j][1]:
#                                found_field = True
#                        elif field in sub_field_array[j][1]:
#                            found_field = True
#                        if found_field:
#                            ISarray[j][sub_field_indx[j]] = total_field_indx
#                            sub_field_indx[j] += 1
#                            total_field_indx += 1
#                            break
#                    if not found_field:
#                        raise ValueError("field ID %d not found in fieldsplit array" %field)
#                    else:
#                        break
#        print(sub_field_indx)
#        print(ISarray)
#        exit()
#        ## Iterate through Section Chart ##
#        pstart,pend = self.section.getChart()
#        dofChart = np.arange(pstart,pend)
#        for i in np.nditer(dofChart):
#            for field in full_field_array:
#                ## Check if this DoF is associated with the given field ##
#                numDof = self.section.getFieldDof(i,field)
#
#                ## If it is, find which sub field array it belongs to ##
#                if numDof > 0:
#                    found_field = False
#                    for j in range(num_sub_fields):
#                        if isinstance(sub_field_array[j][1],int):
#                            if field == sub_field_array[j][1]:
#                                found_field = True
#                        elif field in sub_field_array[j][1]:
#                            found_field = True
#                        if found_field:
#                            ISarray[j][sub_field_indx[j]] = total_field_indx
#                            sub_field_indx[j] += 1
#                            total_field_indx += 1
#                            break
#                    if not found_field:
#                        raise ValueError("field ID %d not found in fieldsplit array" %field)
#                    else:
#                        break
        if self.log_level >= 4:
            timer_iterSecChart.stop()

        if self.log_level >= 4:
            timer_PETScIS = df.Timer("pFibs: Setup Solver Options - set_fieldsplit - create PCFieldSplit - extract_IS - create PETSc IS")
        ## Global offsets communicated via MPI_Scan ##
        goffset = comm.scan(total_field_indx) - total_field_indx

        # Create PETSc IS #
        agg_IS = []
        for i in range(num_sub_fields):
            ISset = np.add(ISarray[i],goffset)
            agg_IS.append((sub_field_array[i][0],PETSc.IS().createGeneral([ISset])))

        if self.log_level >= 4:
            timer_PETScIS.stop()

        if self.log_level >= 3:
            timer.stop()
        return agg_IS

    ## Set PETScOptions parameters ##
    def _set_petsc_options(self,prefix,solver_params):
        for key in solver_params:
            if type(solver_params[key]) is bool:
                if solver_params[key] is True:
                    df.PETScOptions.set(prefix+key)
            elif solver_params[key] is not None:
                df.PETScOptions.set(prefix+key,solver_params[key])
        
