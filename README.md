# pFiBS: A parallel FEniCS implementation of Block Solvers

## Simple Description:

This software is a Python package designed to act as an interface between FEniCS and PETSc to facilitate the construction and application of parallel block solvers/preconditioners. The original intent of this software was to help enable Battery and Wind Farm simulations for use on high performance computing systems. The code is written in Python and uses the petsc4py module to access the more advance features of the PETSc Krylov Solver. In essence, pFiBS is simply an interface to make accessing these features more streamlined. Additionally, pFiBS also provides a template for building custom Python based preconditioning algorithms.  

## Installation:

In order to use pFiBS, the python version FEniCS 2017.2.0 or later must be installed and compiled with the PETSc linear algebra backend. If those critera are met, then pFiBS can be installed by downloading the source files from the GitHub (https://github.nrel.gov/jallen/pfibs) and running the command 
```
pip install -e .
```
in the root source folder. 

## How to Use: 

###### Create the Solver:

Once installed, pFiBS can be accessed by adding `from pfibs import *` to the python code. The solver portion of pFiBS contains two main classes: `LinearVariationalBlockProblem`, and `NonLinearVariationalBlockProblem`. These two classes operate very similarly to FEniCS's built-in `LinearVariationalProblem`, and `NonLinearVariationalProblem`. To create the linear solver use:
```
solver = LinearVariationalBlockProblem(a, L, u, **kwargs)
```
or for the nonlinear solver:
```
solver = NonLinearVariationalBlockProblem(J, F, u, **kwargs)
```
where 'a' is the bilinear form, 'L' is the linear form, 'J' is the Jacobian, 'F' is the linearized RHS, and 'u' is the solution Function. This class will attempt to solve the linear system a(u)=L. Additionally there are several keyword arguments:

- **bcs:** provides a list of boundary conditions to apply to the system. Must be a list of DirichletBC objects, defaults to None.
- **block_structure:** provides the user with the ability to customize how to set up the blocks. Data structure explained below, defaults to None.
- **adjoint:** indicates whether or not to use the dolfin-adjoint solver. Must be a boolean, defaults to False. 
- **annotate:** if the dolfin-adjoint solver is used, indicates whether or not the solve should be annotated. Must be a boolean, defaults to False. 
- **ident_zeros:** Indicates that zero rows of A be replaced by identity. Must be a boolean, defaults to False. 

The most complicated of these keyword arguments is block_structure. Normally, FEniCS uses a FunctionSpace that contains sub FunctionSpaces. If block_structure is not set, the default behavior is to create a block from each sub FunctionSpace on the first level. For example, if the FunctionSpace is composed of a VectorElement and a FiniteElement and if block_structure is not specified, then pFiBS will create two blocks, one for the VectorElement and one for the FiniteElement. The default naming for these blocks will be 0 and 1. 

However, consider a FunctionSpace was composed of 3 sub FunctionSpaces all of which were FiniteElement. If the user wanted to create a block structure that placed the first two subspaces into a block and the last subspace into a separate block, the user would define:
```
block_structure = [['u',[0,1]],['p',[2]]]
```
This tells pFiBS that there will be two blocks. The first block will be named 'u' and contains the DOFs from subspaces 0 and 1. The second block is named 'p' and contains the DOFs from subspace 2. 

###### Set Solver/Preconditioner Type:

After the solver is created, there are two main options to set, KSP type and PC type for all of the blocks and overall solver. These don't need to be set and will default to the PETSc defaults of gmres and ilu. To set the KSP type to 'cg' for the whole system use:
```
solver.setKSPType('cg')     or     PETScOptions.set('ksp_type','cg')
```
where 'cg' can be replaced by any desired KSP type. The KSP type of the first block can be set to 'cg' with:
```
solver.setKSPType(0,'cg')     or     PETScOptions.set('fieldsplit_0_ksp_type','cg')
```
Again, 'cg' is just an example. If the first block had been named 'u' for example, these commands change to:
```
solver.setKSPType('u','cg')     or     PETScOptions.set('fieldsplit_u_ksp_type','cg')
```
The precondtioner can be set in the same way except replacing KSPType for PCType:
```
solver.setPCType('u','hypre')     or     PETScOptions.set('fieldsplit_u_pc_type','hypre')
```

Make sure all calls to `PETScOptions.set()` occur before any call to `setKSPType/setPCType`. Additionally, `solver.setFromOptions()` must be called after all the `PETScOptions.set()` and `setKSPType/setPCType` calls.

Currently, there is limited support for setting up Schur Complement preconditioners. This can be setup using the `solver.setSchur()` function. 

###### Solve:

Finally, after everything is setup, the system can be solved using:
```
solver.solve()
```
See demos for further details.
