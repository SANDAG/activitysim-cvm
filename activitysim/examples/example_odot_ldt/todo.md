# remaining things to do

<ul>
    <li>Clean up some comments/code in the models</li>
    <li>Linting</li>
    <li>Verify the specifications, especially for internal mode/destination choice</li>
    <ul>
        <li>Not entirely sure for internal mode choice if AIR_FAR is fare or not</li>
        <li>Also no entirely sure if cost skims are in cents (it is very likely they are)</li>
    </ul>
    <li>Final validation/testing of activitysim model</li>
    <ul>
        <li>Most models seem to line up with the Java model roughly, aside from the tour gen model for households/workrelated trips, which overestimate/underestimate respectively</li>
        <li>The internal destination choice distances also seem to drop off much more sharply in asim</li>
        <li>Seems to be due to some specification difference (see notebooks in the script folder)</li>
        <li>Some edge cases may break the LDT model (e.g., a subsection of trips have no members for a pd.concat or some constants.yaml fields not defined--needed for some specifications)</li>
    </ul>
    <li>Rename/standardize the input files (e.g., make sure all land use inputs are the current land_use_final file and get rid of all older versions)</li>
    <ul>
        <li>Currently, the canonical inputs are land_use_final.csv, persons_fixed_cdap.csv, households.csv, and skims.omx</li>
        <li>Can make the land_use_final with the fix_occup.ipynb notebook; persons_fixed_cdap can be found on the dropbox</li>
    </ul>
    <li>Make the internal destination choice more robust<li>
    <ul>
        <li>Can the -99999 hard coded things for 50 mile guarantee be made more elegant?</li>
        <li>Minimum time that must be spent in a destination newly added & currently set to 10 minutes, doesn't make sense to drive 59 minutes one way and spend 2 minutes total ina destination</li>
        <li>To guarantee an internal destination, currently have a -9999999 term in the size term specification -- can this be done in a better way?</li>
    </ul>
    <li>Make the longdist_trips index non-dependent on sample size--i.e., define it like with longdist_tours</li>
    <li>Make hte process of extracting the complete car skims for time/distance more elegant, currently done with a lot of abstraction breaking & settings in yaml files to specify keys</li>
</ul>