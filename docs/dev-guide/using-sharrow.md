
# Using Sharrow

This page will walk through an exercise of running a model with `sharrow`.


## Getting the Code

We'll assume that activitysim and sharrow have been installed in editable
mode, per the [developer install instructions](Developer Installation).

The code to implement `sharrow` in `activitysim` is in pull request 
(#542)[https://github.com/ActivitySim/activitysim/pull/542] 

```{sh}
cd activitysim
gh pr checkout 542
cd ..
```

## Create / install the MTC test scale example

Testing with sharrow requires two steps: test mode and production mode.

In test mode, the code is run to compile all the spec files and 
ascertain whether the functions are working correctly.

```{sh}
activitysim create -e example_mtc -d example_mtc_mini
cd example_mtc_mini
activitysim run -c configs_sh_compile -c configs -d data -o output
```


## Create / install the MTC full scale example

Running the bigger models with sharrow now requires three steps: 
compiling, chunk training, and production.

```{sh}
activitysim create -e example_mtc -d example_mtc_full
cd example_mtc_full
activitysim run -c configs_sh_compile -c configs -d data -o output
```

Then go into `configs_chunktrain/settings.yaml` and adjust chunk sizes, processors, etc.

    households_sample_size: 40000
    chunk_size: 40_000_000_000
    num_processes: 8

before then running the training cycle:

```{sh}
activitysim run -c configs_sh -c configs_chunktrain -c configs -d data -o output
```


Lastly, do the same for the production cycle:

```{sh}
activitysim run -c configs_sh -c configs_production -c configs -d data -o output
```
