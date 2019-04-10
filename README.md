# PFIBS: a Parallel FEniCS Implementation of Block Solvers

## Simple Description:

This software is a Python package designed to act as an interface between FEniCS and PETSc to facilitate the construction and application of parallel block solvers/preconditioners. The original intent of this software was to help enable Battery and Wind Farm simulations for use on high-performance computing systems. The code is written in Python and uses the petsc4py module to access the more advanced features of the PETSc Krylov Solver. In essence, pFibs is merely an interface to make accessing these features more streamlined. Additionally, pFibs also provides a template for building custom Python-based preconditioning algorithms.

## Installation:

PFibs needs the python version of FEniCS >2018.1.0 compiled with the PETSc linear algebra backend. To install pFibs download the source files from the GitHub (https://github.com/NREL/pfibs) and run the following command in the project root folder.
```
pip install -e .
```

WARNING: pFibs may not work with an older version of FEniCS. If you are using FEniCS 2017.2.0, only the `LinearBlockSolver` class described below works.

## Basic usage: 

Add `from pfibs import *` into your existing FEniCS code. Leveraging pFibs to solve your variational formulation involves a four-step process.

### Step 1 - Create a block problem:

The first step is to define the block problem using the class `BlockProblem`. It can be used for both linear and nonlinear problems and operates very similarly to FEniCS's built-in `LinearVariationalProblem` and `NonLinearVariationalProblem` classes. To define the block problem, use:
 ```
 problem = BlockProblem(a, L, u)
 ```
where `a` is the bilinear form (or Jacobian for nonlinear problems), `L` is the linear form (or residual for nonlinear problems), and `u` is the solution Function. The `BlockProblem` can take several possible keyword arguments:
 ```
 problem = BlockProblem(a, L, u, bcs=[], aP=None, ident_zeros=False)
 ```
- **bcs:** A list of boundary conditions to apply to the system. Must be a list of DirichletBC objects.
- **aP:** A bilinear form of the preconditioner. If not provided, `a` is used as the preconditioner.

### Step 2 - Add fields:

The next two steps describe how to create a block structure for the problem. As a backend for customizing different block field assignments and field-splitting techniques, pFibs uses PETSc's Field Split Preconditioner (see https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCFIELDSPLIT.html). The user needs to assign subFunctionSpaces to individual blocks through the `field` function within the `BlockProblem` class:
```
problem.field(name,id,solver={})
```
where 'name' is a user-defined string for this block field, `id` is the index of the desired subFunctionSpace, and `solver` is an optional dictionary of PETSc command line options used to construct the solver for this block. 

#### Advanced field assignment
The user has the option to assign several subFunctionSpaces to an individual block field. FEniCS FunctionSpace may contain several subFunctionSpaces. By default, pFibs creates a block from each subFunctionSpace. For example, if the FunctionSpace for the Stokes equation is composed of a VectorElement and a FiniteElement, the `id` for the velocity `u` and pressure `p` is 0 and 1, respectively:
```
problem.field('u',0)
problem.field('p',1)
```
Alternatively, consider a FunctionSpace composed of 3 sub FunctionSpaces of the type FiniteElement. To create a block structure that places the first two subFunctionSpaces into one block and the last subFunctionSpace into a separate block, the user would define:
```
problem.field('u',[0,1])
problem.field('p',2)
```
Declaring `id` as a list tells pFibs that this particular block field contains multiple subFunctionSpaces. In the example above, the first block is named 'u' and contains the DOFs from subFunctionSpaces 0 and 1. The second block is named `p` and contains the DOFs from subFunctionSpace 2.

Now suppose the user has two VectorFunctionSpaces and wants the organize the block fields by dimension, i.e., x- variables belong to the `x` block  field and y- variables to the `y` block field. Thus, the field functions can be invoked as:
```
problem.field('x',[[0,0],[1,0]])
problem.field('y',[[0,1],[1,1]])
```
If pFibs detects a list within a list, the inner list should contain two numbers. The first number corresponds to the sub FunctionSpace index, and the second number corresponds to the sub-subFunctionSpace index. In the above example:
- [0,0]: x-component of the first VectorFunctionSpace
- [1,0]: x-component of the second VectorFunctionSpace
- [0,1]: y-component of the first VectorFunctionSpace
- [1,1]: y-component of the second VectorFunctionSpace

#### Configuring the PETSc solver
The default KSP and PC types for each field are `gmres` and `ilu` respectively. The user can override these by providing their PETSc solver options. For example, if the user wants to solve the velocity field using the conjugate gradient and algebraic multigrid combination with a relative tolerance of 1e-7, the `solver` dictionary should be:
```
solver = {
    'ksp_type': 'cg',
    'pc_type': 'hypre',
    'ksp_rtol': 1e-7
}
```
Other useful options to diagnose the performance of the iterative block solver are:
```
solver = {
    'ksp_type': 'cg',
    'pc_type': 'hypre',
    'ksp_rtol': 1e-7,
    'ksp_monitor_true_residual': True,
    'ksp_converged_reason': True,
    'ksp_view': True
}
```
See https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetFromOptions.html for more possible solver options.

### Step 3 - Add splits:

After defining all the block fields, the user can complete the creation of the block structure by assigning the individual fields to a split. The assignment happens through the `split` function within the `BlockProblem` class:
```
problem.split(name,fields,solver={})
```
where `name` is a user-defined string for this block split, `fields` is a list containing at least two different fields, and `solver` is an optional dictionary of PETSc command line options used to construct the solver for this field split. The complete field split for the Stokes equation may look something like:
```
problem.field('u',0)
problem.field('p',1)
problem.split('s',['u','p'])
```
When `problem.split()` is invoked without the user invoking any calls to `problem.field()`, pFibs automatically generates the fields across subFunctionSpaces and sets each field's name to its corresponding subFunctionSpace index.

#### Nested fieldsplits 
PFibs can perform a nested field split if `fields` contains another split. For example, the block structure for a two-phase flow can look something like:
```
problem.field('u1',0)
problem.field('p1',1)
problem.field('u2',2)
problem.field('p2',3)
problem.split('s1',['u1','p1'])
problem.split('s2',['u2','p2'])
problem.split('s3',['s1','s2'])
```
The above setup enables the user to customize individual solvers for each of the different phases and physical fields. The very last call to `problem.split()` should be the outer most split.

#### Configuring the PETSc fieldsplit solver

The `pc_type` of each split is set to `fieldsplit` and should not be changed. Instead, the user can customize the field split type. For example, the following options set the Schur complement approach with upper factorization:
```
solver = {
    'ksp_type': 'gmres',
    'pc_fieldsplit_type': 'schur', # choice of additive, multiplicative (default), symmetric_multiplicative, schur
    'pc_fieldsplit_schur_fact_type': 'upper', # choice of diag, lower, upper, full (default)
    'pc_fieldsplit_schur_precondition': 'selfp', # choice of a11 (default), selfp, user
}
```

The default KSP and PC types for field split are `gmres` and `multiplicative` respectively.

### Step 4 - Create a block solver:

Finally, the block problem should be hooked up to either a linear or a nonlinear solver. If the problem is linear, use `LinearBlockSolver`:

```
solver = LinearBlockSolver(problem, options_prefix="", solver={}, ctx={})
```
This class attempts to solve the linear system `a(u)=L`. For nonlinear problems use `NonlinearBlockSolver`:

```
solver = NonlinearBlockSolver(problem, options_prefix="", solver={}, ctx={})
```
In this class, FEniCS `NewtonSolver` class solves the linear system `J(u) = F` and updates the solution until `F` converges. Below is a description of all the optional keyword arguments:

- **options_prefix:** Gives the KSP solver a unique prefix. Necessary if more than one block solver is used in the code.
- **solver:** Dictionary of PETSc command line options used to construct the entire block solver. If set, it will override all solvers set in the 'problem.field()' and 'problem.split()' functions. Primarily used for single-field problems.
- **ctx:** Application context for custom Python preconditioning

To solve the system call:
```
solver.solve()
```

### Note about PETScOptions()

While using pFibs, the user should pass all PETSc command line options in via 'solver' dictionaries. Although FEniCS/DOLFIN has the `PETScOptions()` functionality, we strongly recommend not using it directly as this may interfere with the pFibs backend. 

## Custom Python preconditioning

One salient feature of pFibs is the ability to build custom Python-based preconditioning algorithms. The class `PythonPC` is pFibs' base class for providing a custom preconditioner for an individual field. To use it, the `solver` for the particular field must have the following solver options:
```
solver = {
    'pc_type': 'python',
    'pc_python_type': 'pfibs.PythonPC'
}
```
And the `ctx` dictionary must have the following:
```
ctx = {
    'aP': aP,
    'bcs_aP': bcs_aP, # Optional
    'solver': pythonpc_solver # Optional
}
```
where `aP` is the (required) bilinear form of the preconditioner provided by the user, `bcs_aP` is the optional DirichletBC, and `pythonpc_solver` is a dictionary containing PETSc command line options for the solver inside.

`PythonPC` also serves as a template for building more sophisticated preconditioning algorithms. Users can create their own PythonPC's by inheriting the `PythonPC` class and overloading key member functions like `initialize()`, `update()`, and `apply()`. Parameters need to construct this custom preconditioner can be passed in through the application context 'ctx' dictionary.

#### Pressure-Convection-Diffusion (PCD) solver

The `PCDPC` class is one example of how users can utilize inheriting from the 'PythonPC' class and create a more sophisticated preconditioner. To use it, the user can provide these options  to the pressure field:
```
solver = {
    'pc_type': 'python',
    'pc_python_type': 'pfibs.PCDPC'
}
```
And the `ctx` dictionary should have the following:
```
ctx = {
    'nu': nu,
    'vp_spaces': vp_spaces,
    'bcs_kP': bcs_kP, # Optional
    'solver': pcd_solver # Optional
}
```
where `nu` is the viscosity, `vp_spaces` is a list containing the sub FunctionSpace indices for the velocity and pressure, `bcs_kP` is the optional DirichletBC needed for the stiffness matrix, and `pcd_solver` is a dictionary containing PETSc command line options for the mass and stiffness matrix solvers inside. If the user wants to customize the mass and stiffness matrix solvers, `pcd_solver` should look something like:
```
pcd_solver = {
    ## parameters for the mass matrix ##
    'mP_ksp_type': 'gmres',
    'mP_pc_type': 'bjacobi',
    ## parameters for the stiffness matrix ##
    'kP_ksp_type': 'gmres',
    'kP_pc_type': 'hypre',
}
```
See the demos for further details.
