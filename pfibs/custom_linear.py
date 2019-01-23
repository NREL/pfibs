## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
from petsc4py import PETSc
from pfibs.app_ctx import AppCtx

## Check if Dolfin Adjoint is installed ##
try:
    import dolfin_adjoint as dfa
    dolfin_adjoint_found = True
except ImportError:
    dolfin_adjoint_found = False

class CustomKrylovSolver(df.PETScKrylovSolver):
    def __init__(self, vbp, options_prefix="",ctx={}):
        super(CustomKrylovSolver,self).__init__()
        
        ## Initialize field information if it hasn't been done already ##
        self.finalize_field = vbp.finalize_field
        if not self.finalize_field:
            vbp.setUpFields()
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
        
        ## Attach DM ##
        self.ksp().setDM(vbp.dm)
        self.ksp().setDMActive(False)

        ## Check application context ##
        if not isinstance(ctx,dict):
            raise TypeError("Solver context must be of type dict()")
        #self.ctx = ctx
        self.ctx = AppCtx(ctx)
        
    def init_solver_options(self):
 
        ## Start the timer ##
        timer = df.Timer("pFibs: Setup Solver Options")

        ## Create PETSc commandline options if necessary ##
        if self.split_0 != "":
            self.set_fieldsplit(self.split_0, self.options_prefix)
        else:
            self.ksp().pc.setType(PETSc.PC.Type.FIELDSPLIT)
        ## Set PETSc commandline options ##
        self.ksp().setFromOptions()

        ## Construct the KSP and PC ##
        #self.ksp().setUp()

        ## Attach application context if necessary ##
        #if self.ctx:
        #    print("STOP")
        #    self.app_ctx(self.ksp())

        ## Stop the timer ##
        timer.stop()
    
    ## Set application context recursively ##
    def app_ctx(self, ksp):
        if ksp.pc.getType() == 'fieldsplit':
            ksp.pc.setUp()
            ksp_list = ksp.pc.getFieldSplitSubKSP()
            if len(ksp_list) == 0:
                raise ValueError("Cannot have zero subKSPs in a fieldsplit")
            for subksp in ksp_list:
                self.app_ctx(subksp)
        else:
            ksp.setAppCtx(self.ctx)
            #_, P = ksp.getOperators()
            #P.setPythonContext(self.ctx)

    ## Set fieldsplit solvers via PETScOptions ##
    def set_fieldsplit(self, split_name, prefix, recursion=False):
        
        ## Obtain block information for this split ##
        block_name = self.block_split[split_name][0]
        block_solvers = self.block_split[split_name][1]
        recursive_split = False
        n = len(block_name)

        ## If this split contains more than one block ##
        if n != 1:
            
            ## Set solver options for this split ##
            df.PETScOptions.set(prefix+"pc_type","fieldsplit")
            for key in block_solvers:
                df.PETScOptions.set(prefix+key,block_solvers[key])
        
        ## List of all fields in total ##
        field_array = []
        
        ## List of all fields per split ##
        sub_field_array = []

        ## Iterate through each split ##
        for i in range(n):

            ## Case 1: split contains a field variable ##
            if block_name[i] in self.block_field:

                ## Assign prefix based on field name ##
                new_prefix = prefix+"fieldsplit_"+str(block_name[i])+"_"
                
                ## Set solver options for the field variable ##
                for key in self.block_field[block_name[i]][2]:
                    df.PETScOptions.set(new_prefix+key,self.block_field[block_name[i]][2][key])
                
                ## Update the lists ##
                sub_field_array.append(self.block_field[block_name[i]][0])
                field_array.append(self.block_field[block_name[i]][0])

            ## Case 2: split contains another split ##
            elif block_name[i] in self.block_split:
                recursive_split = True
                
                ## Obtain list of fields within this split ##
                nested_field_array = self.set_fieldsplit(block_name[i],prefix+"fieldsplit_"+repr(i)+"_",True)
                
                ## Update the lists ##
                if isinstance(nested_field_array,list):
                    field_array.extend(nested_field_array)
                else:
                    field_array.append(nested_field_array)
                sub_field_array.append(nested_field_array)
            else:
                raise ValueError("Block '%s' not found in the field/split dicts" %(block_name[i]))
        
        ## Assigning fields to each split ##   
        if recursion and recursive_split or not recursion:
            list_indx = 0
            for i in range(len(sub_field_array)):
                
                ## Case 1: first split only ##
                if not recursion:
                    field_list = str(sub_field_array[i]).strip("[]")
                
                ## Case 2: Nested splits, rescale numbering ##
                else:
                    if isinstance(sub_field_array[i],list):
                        field_length = len(sub_field_array[i])
                    else:
                        field_length = 1
                    field_list = str([j for j in range(list_indx, list_indx+field_length)]).strip("[]")
                    list_indx += field_length
                df.PETScOptions.set(prefix+"pc_fieldsplit_"+repr(i)+"_fields",field_list)
        
        ## Return list of all fields in this split ##
        return field_array

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


