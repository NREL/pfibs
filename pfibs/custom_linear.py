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
        
        ## Obtain DM and block problem ##
        self.dm = vbp.dm
        self.vbp = vbp
        
        ## Check application context ##
        if not isinstance(ctx,dict):
            raise TypeError("Solver context must be of type dict()")
        self.ctx = ctx
        
    def init_solver_options(self):
 
        ## Start the timer ##
        timer = df.Timer("pFibs: Setup Solver Options")
        
        ## Attach DM and application context ##
        self.ksp().setDM(self.dm)
        self.ksp().setDMActive(False)
        self.ksp().setAppCtx(self.ctx)

        ## Set fieldsplit if add_split() not utilized ##
        if self.split_0 == "" and self.num_fields > 1:
            self.ksp().pc.setType(PETSc.PC.Type.FIELDSPLIT)

        ## Setup the PC ##
        self.ksp().setUp()

        ## Construct fieldsplits ##
        if self.split_0 != "" and self.num_fields > 1:
            self.create_fieldsplit(self.options_prefix,self.split_0,self.ksp(),True)
        
        ## Set PETSc commandline options ##
        self.ksp().setFromOptions()

        ## Stop the timer ##
        timer.stop()
    
    ## Create fieldsplit ##
    def create_fieldsplit(self, prefix, split_name, subKSP, setpc=True):
        
        ## Obtain block information for this split ##
        block_name = self.block_split[split_name][0]
        block_solvers = self.block_split[split_name][1]
        n = len(block_name)

        ## List of all fields in total ##
        field_array = []
        
        ## List of all fields per split ##
        sub_field_array = []
        

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
                nested_field_array = self.create_fieldsplit("",block_name[i],subKSP,False)
                
                ## Update the lists ##
                if isinstance(nested_field_array,list):
                    field_array.extend(nested_field_array)
                else:
                    field_array.append(nested_field_array)
                sub_field_array.append((str(block_name[i]),nested_field_array))
            else:
                raise ValueError("Block '%s' not found in the field/split dicts" %(block_name[i]))
        
        ## Step 2: create PCFieldSplit ##
        if setpc:
            
            ## Set the fieldsplit ##
            subKSP.pc.setType(PETSc.PC.Type.FIELDSPLIT)

            ## Is it a schur complement ##
            use_schur = self.block_split[split_name][1].get("pc_fieldsplit_type",False)
            if use_schur == 'schur':
                subKSP.pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            
                       
            ## Assign fields to the splits ##
            assignIS = self.vbp.subIS(sub_field_array,field_array)
            subKSP.pc.setFieldSplitIS(*assignIS)
            
            ## Set solver parameters for split ##
            self.set_fieldsplit_options(prefix,block_solvers)
            
            ## Set solver parameters for individual fields if necessary ##
            for i in range(n):
                if sub_field_array[i][0] in self.block_field:
                    # Assign prefix based on field name ##
                    new_prefix = prefix+"fieldsplit_"+sub_field_array[i][0]+"_"
                    self.set_fieldsplit_options(new_prefix,self.block_field[sub_field_array[i][0]][2])
            
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
                    self.create_fieldsplit(prefix+"fieldsplit_"+sub_field_array[i][0]+"_",
                            sub_field_array[i][0],subSubKSP[i],True)
        
        ## Return list of all fields in this split ##
        return field_array

    ## Set the PETScOptions for fieldsplit parameters ##
    def set_fieldsplit_options(self,prefix,solver_params):
        for key in solver_params:
            if type(solver_params[key]) is bool:
                if solver_params[key] is True:
                    df.PETScOptions.set(prefix+key)
            elif solver_params[key] is not None:
                df.PETScOptions.set(prefix+key,solver_params[key])
        

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
