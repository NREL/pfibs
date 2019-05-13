import dolfin as df
import dolfin_adjoint as dfa
import pfibs as pf
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from fenics_adjoint.solving import SolveBlock

class BlockProblem(pf.BlockProblem):
    @no_annotations
    def __init__(self, *args, **kwargs):
        super(BlockProblem, self).__init__(*args, **kwargs)

        self._ad_A = args[0]
        self._ad_b = args[1]
        self._ad_u = args[2]
        self._ad_bcs = kwargs.get("bcs",[])
        self._ad_args = args
        self._ad_kwargs = kwargs

        self.block_helper = BlockSolveBlockHelper()

class LinearBlockSolver(pf.LinearBlockSolver):
    @no_annotations
    def __init__(self,problem,*args, options_prefix=None, comm=None, **kwargs):
        super(LinearBlockSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, **kwargs):
        annotate = annotate_tape()
        if annotate:
            block_helper = BlockSolveBlockHelper()
            tape = get_working_tape()
            problem = self._ad_problem


#            sb_kwargs = SolveBlock.pop_kwargs(kwargs)
            block = LinearBlockSolveBlock(problem._ad_A==problem._ad_b,
                                             problem._ad_u,
                                             problem._ad_bcs,
                                             block_helper=block_helper,
                                             block_field = self._ad_problem.block_field,
                                             block_split = self._ad_problem.block_split)
            tape.add_block(block)

        with stop_annotating():
            out = super(LinearBlockSolver, self).solve()

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out

class NonlinearBlockSolver(pf.NonlinearBlockSolver):
    @no_annotations
    def __init__(self,problem,*args, options_prefix=None, comm=None, **kwargs):
        super(NonlinearBlockSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, **kwargs):
        annotate = annotate_tape()
        if annotate:
            block_helper = BlockSolveBlockHelper()
            tape = get_working_tape()
            problem = self._ad_problem


#            sb_kwargs = SolveBlock.pop_kwargs(kwargs)
            block = NonlinearBlockSolveBlock(problem._ad_b == 0,
                                             problem._ad_u,
                                             problem._ad_bcs,
                                             block_helper=block_helper,
                                             problem_J = problem._ad_A,
                                             block_field = self._ad_problem.block_field,
                                             block_split = self._ad_problem.block_split)
            tape.add_block(block)

        with stop_annotating():
            out = super(NonlinearBlockSolver, self).solve()

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out

class BlockSolveBlockHelper(object):
    def __init__(self):
        self.forward_solver = None
        self.adjoint_solver = None

    def reset(self):
        self.forward_solver = None
        self.adjoint_solver = None

class LinearBlockSolveBlock(SolveBlock):
    def __init__(self, *args, **kwargs):
        super(LinearBlockSolveBlock, self).__init__(*args, **kwargs)
        self.block_field = kwargs.pop("block_field")
        self.block_split = kwargs.pop("block_split")
        self.block_helper = kwargs.pop("block_helper")

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        solver = self.block_helper.forward_solver
        if solver is None:
            problem = pf.BlockProblem(lhs, rhs, func, bcs=bcs)
            for key, value in self.block_field.items():
                problem.field(key, value[1], solver=value[2])
            for key, value in self.block_split.items():
                problem.split(key, value[0], solver=value[1])
            solver = pf.LinearBlockSolver(problem)
            self.block_helper.forward_solver = solver
        solver.solve()
        return func

    def _assemble_and_solve_adj_eq(self, dFdu_form, dJdu):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()

        solver = self.block_helper.adjoint_solver
        if solver is None:

            adj_sol = df.Function(self.function_space)

            adjoint_problem = pf.BlockProblem(dFdu_form, dJdu, adj_sol, bcs=bcs)
            for key, value in self.block_field.items():
                adjoint_problem.field(key, value[1], solver=value[2])
            for key, value in self.block_split.items():
                adjoint_problem.split(key, value[0], solver=value[1])
            solver = pf.LinearBlockSolver(adjoint_problem)
            self.block_helper.adjoint_solver = solver

############
            ## Assemble ##
            rhs_bcs_form = df.inner(df.Function(self.function_space),dFdu_form.arguments()[0]) * df.dx
            A_, _ = df.assemble_system(dFdu_form, rhs_bcs_form, bcs)
            A = df.as_backend_type(A_)

            solver.linear_solver.set_operators(A,A)
            solver.linear_solver.init_solver_options()

        [bc.apply(dJdu) for bc in bcs]
        b = df.as_backend_type(dJdu)
        ## Actual solve ##
        its = solver.linear_solver.solve(adj_sol.vector(),b)
###########

        adj_sol_bdy = dfa.compat.function_from_vector(self.function_space, dJdu_copy - dfa.compat.assemble_adjoint_value(df.action(dFdu_form, adj_sol)))
        return adj_sol, adj_sol_bdy

class NonlinearBlockSolveBlock(SolveBlock):
    def __init__(self, *args, **kwargs):
        super(NonlinearBlockSolveBlock, self).__init__(*args, **kwargs)
        self.nonlin_problem_J = kwargs.pop("problem_J")
        self.block_field = kwargs.pop("block_field")
        self.block_split = kwargs.pop("block_split")
        self.block_helper = kwargs.pop("block_helper")


        for coeff in self.nonlin_problem_J.coefficients():
            self.add_dependency(coeff, no_duplicates=True)

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        solver = self.block_helper.forward_solver
        if solver is None:
            J = self.nonlin_problem_J
            if J is not None:
                J = self._replace_form(J, func)
            problem = pf.BlockProblem(J, lhs, func, bcs=bcs)
            for key, value in self.block_field.items():
                problem.field(key, value[1], solver=value[2])
            for key, value in self.block_split.items():
                problem.split(key, value[0], solver=value[1])
            solver = pf.NonlinearBlockSolver(problem)
            self.block_helper.forward_solver = solver
        solver.solve()
        return func

    def _assemble_and_solve_adj_eq(self, dFdu_form, dJdu):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()

        solver = self.block_helper.adjoint_solver
        if solver is None:

            adj_sol = df.Function(self.function_space)

            adjoint_problem = pf.BlockProblem(dFdu_form, dJdu, adj_sol, bcs=bcs)
            for key, value in self.block_field.items():
                adjoint_problem.field(key, value[1], solver=value[2])
            for key, value in self.block_split.items():
                adjoint_problem.split(key, value[0], solver=value[1])
            solver = pf.LinearBlockSolver(adjoint_problem)
            self.block_helper.adjoint_solver = solver

############
            ## Assemble ##
            rhs_bcs_form = df.inner(df.Function(self.function_space),dFdu_form.arguments()[0]) * df.dx
            A_, _ = df.assemble_system(dFdu_form, rhs_bcs_form, bcs)
            A = df.as_backend_type(A_)

            solver.linear_solver.set_operators(A,A)
            solver.linear_solver.init_solver_options()

        [bc.apply(dJdu) for bc in bcs]
        b = df.as_backend_type(dJdu)
        ## Actual solve ##
        its = solver.linear_solver.solve(adj_sol.vector(),b)
###########

        adj_sol_bdy = dfa.compat.function_from_vector(self.function_space, dJdu_copy - dfa.compat.assemble_adjoint_value(df.action(dFdu_form, adj_sol)))
        return adj_sol, adj_sol_bdy
