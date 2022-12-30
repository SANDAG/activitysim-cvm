# Ohio DOT Long Distance ActivitySim Implementation

Example data and configuration files for the long distance travel models.

Data files based on the 2010_EC Ohio DOT model of record are stored in the
activitysim_resources repo.  They can be installed automatically with
the `activitysim create` command:

```
activitysim create -e prototype_odot_ldt_full -d .
```

Note employment data in this published dataset is modified from the original
proprietary data used by Ohio DOT in the actual model of record.

## Suggested Install Process for Development

ActivitySim has a lot of dependencies. It’s easiest and fastest to install them
using a package manager like conda. There’s a faster version called
[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).  If you've
already installed conda but not mamba, you can add it after the fact, but you
should only install mamba in the base environment. If you install mamba itself
in other environments, it will not function correctly. If you’ve got an existing
conda installation and you want to install mamba into it, you can install mamba
into the base environment like this:

```
conda update conda -n base
conda install -n base -c conda-forge mamba
```

Once you've got mamba, you can install all the other dependencies in a single
workspace directory (rename "workspace" to something else if you like):

```
mkdir workspace
cd workspace
mamba env create -p ASIM-LDT --file https://raw.githubusercontent.com/camsys/activitysim/pydata-docs/conda-environments/activitysim-dev-2.yml
conda activate ./ASIM-LDT
git clone https://github.com/ActivitySim/sharrow.git
python -m pip install -e ./sharrow
git clone https://github.com/camsys/activitysim.git
cd activitysim
git switch longdist
python -m pip install -e .
```


# Running

```
conda activate ASIM-LDT
activitysim run -c configs -d data -o output --ext extensions
```

# Random issues

- in ldt_trip_generation_houeseholds, the ODOT model specification includes person-level variables in a household-level choice model.  I'm not sure why we did that or how that works.  Need to go back and look at ODOT code to figure it out.
- In LDT trip generation for persons, is there a more elegant way to segment by trip purpose.
