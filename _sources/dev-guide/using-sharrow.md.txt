
# Using Sharrow

This page will walk through an exercise of running a model with `sharrow`.


## Getting the Code

We'll assume that activitysim and sharrow have been installed in editable
mode, per the [developer install instructions](install.md).

The code to implement `sharrow` in `activitysim` is in pull request 
[#542](https://github.com/ActivitySim/activitysim/pull/542). 

```sh
cd activitysim
gh pr checkout 542  # use `gh auth login` first if needed
cd ..
```

## MTC Test Example

Testing with sharrow requires two steps: test mode and production mode.

In test mode, the code is run to compile all the spec files and 
ascertain whether the functions are working correctly.  Production mode
can then just run the pre-compiled functions with sharrow, which is much 
faster.

You can run both, plus the legacy ActivitySim, all together in one workflow:

```sh
activitysim workflow sharrow-contrast/mtc_mini
```


## MTC Full Example

Running the bigger models with sharrow now requires three steps: 
compiling, chunk training, and production.

```sh
activitysim create -e example_mtc_full -d example_mtc_full
cd example_mtc_full
activitysim run -c configs_sh_compile -c configs -d data -o output
```

Then go into `configs_chunktrain/settings.yaml` and adjust chunk sizes, processors, etc.

    households_sample_size: 40000
    chunk_size: 40_000_000_000
    num_processes: 8

before then running the training cycle:

```sh
activitysim run -c configs_sh -c configs_chunktrain -c configs -d data -o output
```


Lastly, do the same for the production cycle:

```sh
activitysim run -c configs_sh -c configs_production -c configs -d data -o output
```



## ARC Test Example

```sh
activitysim create -e example_arc -d example_arc_mini
cd example_arc_mini
activitysim run -c configs_sh_compile -c configs -d data -o output
```

The same code can then be run without the compile/test flags.

```sh
activitysim run -c configs_sh -c configs -d data -o output
```

