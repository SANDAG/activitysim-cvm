
# Developer Installation

Installing ActivitySim as a developer is almost as easy as just using it,
but making some tweaks to the processes enables live code development and 
testing.

## Package Manager

ActivitySim has a lot of dependencies.  It's easiest and fastest to install
them using a package manager like conda. There's a faster version called
[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).
Depending on your security settings, you might need to install in a 
container like docker, instructions for that are coming soon.

Note that if you are installing `mamba`, you only should install `mamba` 
in the *base* environment. If you install `mamba` itself in other environments, 
it will not function correctly.

While you are at it, if you are a Jupyter user you might want to also install 
`nb_conda_kernels` in your base conda environment alongside any other `jupyter` 
libraries: 

```sh
mamba install -n base nb_conda_kernels -c conda-forge
```

This will ensure your development environments are selectable as kernels in
Jupyter Notebook/Lab/Etc.
 
## Environment

It's convenient to start from a completely clean conda environment 
and git repository.  Assuming you have `mamba` installed, you can do so 
by starting where ActivitySim is not yet cloned (e.g. in an empty 
directory) and running:

```sh
mamba create -n ASIM-DEV python=3.9 git gh -c conda-forge --override-channels
conda activate ASIM-DEV
gh auth login   # <--- (only needed if gh is not logged in)
gh repo clone ActivitySim/activitysim
gh repo clone ActivitySim/sharrow
cd activitysim
```

Note the above commands will clone both ActivitySim and sharrow, so that
you can potentially edit and commit changes to both libraries.
    
## Dependencies

You can install all the dependencies except `sharrow` and `activitysim`
itself using the pre-made environment definition file here: 

```sh
mamba env update --file=conda-environments/activitysim-dev-2.yml
```

```{important}
If you add to the ActivitySim dependencies, make sure to also update 
the environments in `conda-environments`, which are used for testing 
and development.  If they are not updated, these environments will end 
up with dependencies loaded from *pip* instead of *conda-forge*.
```

Then, use pip to install the last libraries in editable mode, which
will allow your code changes to be reflected when running ActivitySim
in this environment.

```sh
python -m pip install -e ./sharrow
python -m pip install -e ./activitysim
```

Now your environment should be ready to use.  Happy coding!
