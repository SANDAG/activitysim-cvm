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
        <li>Seems to be due to some specification difference (see notebooks in the script folder)</li>
        <li>Some edge cases may break the LDT model (e.g., a subsection of trips have no members)</li>
    </ul>
    <li>Rename/standardize the input files (e.g., make sure all land use inputs are the current land_use_fixed file and get rid of all older versions)</li>
</ul>