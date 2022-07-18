# Ohio DOT Long Distance ActivitySim Implementation

Example data and configuration files for the long distance travel models.  The default setup contains data for a small subset.  data_full is the full data for the 2010_EC model of record that can be provided upon request. 

# Install

As of 7.17.2022, the standard development install methods described in the ActivitySim documentation fail on a Windows computer.  The issue is that pytables 3.7.0 is incompatible with other required packages.  The packages will install, but when pip tries to build the install from the cloned directory, the install of a previous version of pytables will attempt to build the wheel itself.  This requires a number of packages to be pre-existing on windows, and gets messy.  What did work is this:

1. Use conda to install the packages described for the development environment. 
2. Use conda to install pytables 3.6.1: conda install pytables=3.6.1 -c conda-forge
3. Use pip to install the activitysim from the current directory in editable mode (note the order of the . and the -e are different from the ActivitySim documentation): pip install -e .

# Running

Use the batch file, or: 

conda activate asim-dev
activitysim run -c configs -d data_full -o output


# Random issues

- in ldt_trip_generation_houeseholds, the ODOT model specification includes person-level variables in a household-level choice model.  I'm not sure why we did that or how that works.  Need to go back and look at ODOT code to figure it out. 
- LDT trip generation choice model seems to work for households.  Need to add variable to household table. 
- In LDT trip generation for persons, there is an error.  It appears the persons_merged table is empty. 
- In LDT trip generation for persons, is there a more elegant way to segment by trip purpose. 
