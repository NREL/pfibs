## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
from pfibs.custom_linear import CustomKrylovSolver
from pfibs.custom_nonlinear import NLP, NS

## Optionally import dolfin_adjoint ##
try:
    import dolfin_adjoint as dfa 
    dolfin_adjoint_found = True
except ImportError:
    dolfin_adjoint_found = False

class BaseBlockSolver(object):
    def __init__(self, vbp, options_prefix="", solver={}, ctx={}):

        self.a = vbp.a
        self.aP = vbp.aP
        self.L = vbp.L
        self.bcs = vbp.bcs
        self.u = vbp.u
        self.linear_solver = CustomKrylovSolver(vbp,options_prefix=options_prefix,
                                                solver=solver,ctx=ctx)

class LinearBlockSolver(BaseBlockSolver):
    def __init__(self, vbp, options_prefix="", solver={}, ctx={}):
        
        super().__init__(vbp, options_prefix, solver, ctx)

        self.A = df.PETScMatrix()
        self.b = df.PETScVector()

        self.log_level = vbp.log_level

        if self.aP is not None:
            self.P = df.PETScMatrix()

    def assemble(self):
        
        ## Time function execution ##
        timer = df.Timer("pFibs: Assemble")
        timer.start()
        
        if self.log_level >= 1:
            ## Time system assembly ##
            timer1 = df.Timer("pFibs: Assemble - System")

        ## Assembly system of equations ##
        df.assemble_system(self.a,self.L,self.bcs,A_tensor=self.A,b_tensor=self.b)
        if self.log_level >= 1:
            timer1.stop()

        if self.log_level >= 1:
            ## Time preconditioner assembly ##
            timer2 = df.Timer("pFibs: Assemble - Preconditioner")

        ## Assemble preconditioner if provided ##
        if self.aP is not None:
            df.assemble(self.aP,tensor=self.P)
            if isinstance(self.bcs,list):
                for bc in self.bcs:
                    bc.apply(self.P)
            else:
                self.bcs.apply(self.P)
        if self.log_level >= 1:
            timer2.stop()

        # self.A.mat().zeroEntries()
        # self.A.mat().shift(1.0)

        timer.stop()

    def solve(self):
        ## Assemble ##
        self.assemble()

        ## Setup solver operators and options ##
        if self.aP is not None:
            self.linear_solver.set_operators(self.A,self.P)
        else:
            self.linear_solver.set_operators(self.A,self.A)
        self.linear_solver.init_solver_options()
        
        ## Start the timer ##
        timer = df.Timer("pFibs: Solve")

        ## Actual solve ##
        its = self.linear_solver.solve(self.u.vector(),self.b)
        
        ## Stop the timer ##
        timer.stop()

        return its

class NonlinearBlockSolver(BaseBlockSolver):
    def __init__(self, vbp, options_prefix="", solver={}, ctx={}):
        
        ## Time function execution ##
        timer = df.Timer("pFibs: Init nonlinear block solver")
        timer.start()
        
        super().__init__(vbp, options_prefix, solver, ctx)
        self.ident_zeros = vbp.ident_zeros
        self.newton_solver = NS(self.linear_solver)

        self._init_nlp = False
        self.bcs_u = []

        timer.stop()

    def applyBC(self):
        ## Start the timer ##
        timer = df.Timer("pFibs: Apply BCS")

        ## Assembly system of equations ##
        if isinstance(self.bcs,list):
            for bc in self.bcs:
                bc.apply(self.u.vector())
                bc0 = df.DirichletBC(bc)
                bc0.homogenize()
                self.bcs_u.append(bc0)
        else:
            self.bcs.apply(self.u.vector())
            bc0 = df.DirichletBC(bc)
            bc0.homogenize()
            self.bcs_u.append(bc0)

        ## Stop the timer ##
        timer.stop()
    
    def solve(self):
        ## Start the timer ##
        timer = df.Timer("pFibs: Solve")
        
        ## Apply bcs ##
        self.applyBC()

        ## Create nonlinear problem ##
        if not self._init_nlp:
            self.problem = NLP(self.a, self.L, self.aP, bcs=self.bcs_u, ident_zeros=self.ident_zeros, ksp=self.linear_solver.ksp())
            self._init_nlp == True

        ## Actual solve ##
        its, converged = self.newton_solver.solve(self.problem,self.u.vector())

        ## Stop the timer ##
        timer.stop()

        return its, converged
