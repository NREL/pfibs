# PFIBS: a Parallel FEniCS Implementation of Block Solvers

## Simple Description:

This software is a Python package designed to act as an interface between FEniCS and PETSc to facilitate the construction and application of parallel block solvers/preconditioners. The original intent of this software was to help enable Battery and Wind Farm simulations for use on high performance computing systems. The code is written in Python and uses the petsc4py module to access the more advance features of the PETSc Krylov Solver. In essence, pFibs is simply an interface to make accessing these features more streamlined. Additionally, pFibs also provides a template for building custom Python based preconditioning algorithms.  

## Installation:

In order to use pFibs, the python version FEniCS 2017.2.0 or later must be installed and compiled with the PETSc linear algebra backend. If those critera are met, then pFiBS can be installed by downloading the source files from the GitHub (https://github.com/NREL/pfibs) and running the command 
```
pip install -e .
```
in the root source folder. 

## How to Use: 

Add `from pfibs import *` into your existing FEniCS code. Leveraging pFibs to solve your variational formulation is a four step process.

###### Step 1 - (Optional) Define the block structure:

The user has the option to manually define the block structure. Normally, FEniCS uses a FunctionSpace that contains sub FunctionSpaces. The default behavior is to create a block from each sub FunctionSpace on the first level. For example, if the FunctionSpace is composed of a VectorElement and a FiniteElement and if block_structure is not specified, then pFibs will create two blocks, one for the VectorElement and one for the FiniteElement. The default naming for these blocks will be 0 and 1. 

If the user wants to provide different names for these two fields, e.g., fields 'u' and 'p' for the VectorElement and FiniteElement, respectively, the block structure should be defined as:
```
block_structure = [['u',0],['p',1]]
```
This tells pFibs that there will be two blocks. The first block will name the 0-th and 1st sub FunctionSpaces 'u' and 'p' respectively.

Alternatively, consider a FunctionSpace was composed of 3 sub FunctionSpaces all of which were FiniteElement. If the user wanted to create a block structure that placed the first two subspaces into a block and the last subspace into a separate block, the user would define:
```
block_structure = [['u',[0,1]],['p',[2]]]
```
This tells pFiBS that there will be two blocks. The first block will be named 'u' and contains the DOFs from subspaces 0 and 1. The second block is named 'p' and contains the DOFs from subspace 2. 

Now suppose the user has two VectorFunctionSpaces and wants the block solve grouped by dimension i.e., x- variables belong to the 'x' field and y- variables to the 'y' field. Thus, block_structure would be defined as:
```
block_structure = [['x',[[0,0],[1,0]]],['y',[[1,0],[1,1]]]]
```
###### Step 2 - Create the Block Problem:

The second step is to define the block problem using the class `BlockProblem`. It can be used for both linear and nonlinear problems and operates very similarly to FEniCS's built-in `LinearVariationalProblem` and `NonLinearVariationalProblem` classes. To simply define the block problem, use:
 ```
 problem = BlockProblem(a, L, u)
 ```
where 'a' is the bilinear form (or Jacobian for nonlinear problems), 'L' is the linear form (or residual for nonlinear problems), and 'u' is the solution Function. The `BlockProblem` can be invoked with several possible keyword arguments:
 ```
 problem = BlockProblem(a, L, u, bcs=[], block_structure=None, adjoin=False, annotate=False, ident_zeros=False)
 ```
- **bcs:** provides a list of boundary conditions to apply to the system. Must be a list of DirichletBC objects.
- **block_structure:** provides the user with the ability to customize how to set up the blocks.
- **adjoint:** indicates whether or not to use the dolfin-adjoint solver. 
- **annotate:** if the dolfin-adjoint solver is used, indicates whether or not the solve should be annotated.
- **ident_zeros:** Indicates that zero rows of A be replaced by identity. 

###### Step 3 - Set Solver/Preconditioner Type:

After the problem is created, there are two main options to set: KSP type and PC type for all of the blocks and overall solver. These don't need to be set and will default to the PETSc defaults of gmres and ilu. To set the KSP type to 'cg' for the whole system use:
```
problem.setKSPType('cg')     or     PETScOptions.set('ksp_type','cg')
```
where 'cg' can be replaced by any desired KSP type. The KSP type of the first block can be set to 'cg' with:
```
problem.setKSPType(0,'cg')     or     PETScOptions.set('fieldsplit_0_ksp_type','cg')
```
Again, 'cg' is just an example. If the first block had been named 'u' for example, these commands change to:
```
problem.setKSPType('u','cg')     or     PETScOptions.set('fieldsplit_u_ksp_type','cg')
```
The precondtioner can be set in the same way except replacing KSPType for PCType:
```
problem.setPCType('u','hypre')     or     PETScOptions.set('fieldsplit_u_pc_type','hypre')
```

Currently, there is limited support for setting up Schur Complement preconditioners. This can be setup using the `solver.setSchur()` function. 

###### Step 4 - Create the Block Solver:

After everything is setup, the block problem should be hooked up to either a linear or a nonlinear solver. If the problem is linear, use `LinearBlockSolver`:

```
solver = LinearBlockSolver(problem, options_prefix=None)
```
This class will attempt to solve the linear system a(u)=L. If the problem is nonlinear, use `NonlinearBlockSolver`:

```
solver = NonlinearBlockSolver(problem, option_prefix=None)
```
In this class, FEniC's newton solves the linear system J(u) = F and updates the solution until F converges. If multiple block solvers are utilized in the code, 'options_prefix' needs to be set if the PETScOptions is utilized to configure KSP and PC parameters. To solve the system can be solved using:
```
solver.solve()
```
. See demos for further details.
