## Future-proofing for Python3+
from __future__ import print_function

## Import dolfin and numpy and time ##
import dolfin as df
import sys

class NLP(df.NonlinearProblem):
    def __init__(self, a, L, aP, bcs=None, ident_zeros=False, ksp=None):
        
        ## Time function execution ##
        timer = df.Timer("pFibs: Init FEniCS nonlinear problem")
        timer.start()
        
        super(NLP, self).__init__()
        self.L = L
        self.a = a
        self.aP = aP
        self.bcs = bcs
        self.ident_zeros = ident_zeros
        self.set_me = False
        self.ksp = ksp

        timer.stop()
    
    def F(self,b,x):
        ## Start the timer ##
        timer = df.Timer("pFibs: Assemble RHS")

        ## Assemble ##
        df.assemble(self.L, tensor=b)

        for bc in self.bcs:
            bc.apply(b)

        ## Stop Timer ##
        timer.stop()

    def J(self,A,x):
        ## Start the timer ##
        timer = df.Timer("pFibs: Assemble Jacobian")

        ## Assemble ##
        df.assemble(self.a, tensor=A)

        for bc in self.bcs:
            bc.apply(A)

        if self.ident_zeros:
            A.ident_zeros()

        ## Stop Timer ##
        timer.stop()

class NS(df.NewtonSolver):
    def __init__(self, linear_solver):
        
        ## Time function execution ##
        timer = df.Timer("pFibs: Init Newton solver")
        timer.start()
        
        comm = linear_solver.ksp().comm.tompi4py()
        factory = df.PETScFactory.instance()
        super(NS, self).__init__(comm, linear_solver, factory)
        self._solver = linear_solver
    def solver_setup(self, A, P, nlp, iteration):
        
        ## Time function execution ##
        timer = df.Timer("pFibs: Newton solver setup")
        timer.start()
        
        if nlp.aP is not None:
            timer2 = df.Timer("pFibs: Newton solver setup - preconditioner")
            df.assemble(nlp.aP,tensor=P)
            for bc in nlp.bcs:
                bc.apply(P)
            timer2.stop()
        if iteration > 0 or getattr(self, "_initialized", False):
            return
        P = A if P.empty() else P
        self._solver.set_operators(A,P)
        self._initialized = True
        self._solver.init_solver_options()

        timer.stop()

    def solve(self, problem, x):
        
        timer = df.Timer("pFibs: Newton solver solve")
        
        self._problem = problem
        # print(self._solver.ksp().setComputeSingularValues(True))
        r = super(NS, self).solve(problem, x)
        # print(self._solver.ksp().computeExtremeSingularValues())
        # exit()
        del self._problem

        timer.stop()

        return r

    def linear_solver(self):
        return self._solver

