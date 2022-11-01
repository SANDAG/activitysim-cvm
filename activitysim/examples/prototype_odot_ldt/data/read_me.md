gde 3.22.2022
jmh 3.25.2022

Get full data files from Dropbox site. Currently in development. Contact greg.erhardt@uky.edu.

These inputs are derived from the Ohio Statewide Model for use in the ActivitySim LDT
implementation.  They are converted for year 2010 from Baseline_EC Scenario of the
Model of Record (MOR) as of 3.22.2022.  The source folder on the ODOT servers is:

    T:\MOR\osmp\Base\Baseline_EC\t10

The relevant input files are:

land_use_updated.csv

    For use in the ActivitySim framework, we have consolidated data from multiple files
    into a single land_use.csv file with one record for each TAZ (both internal and external).
    The fields are listed below.  For each field, we a description as well as the orignal
    source file and field name.  In this combined file, we have standardized the field
    names to remove spaces, and to be consistent in capitalization and notation.

    Because the ODOT employment data cannot be shared publicly, we have derived
    the equivalent data from LEHD.

    TAZ - TAZ ID (TAZ column in TAZtoAMZ.csv)
    MODELAREA - Binary flag indicating if the TAZ is within the model area (MODELAREA column in TAZtoAMZ.csv)
    CORDONZONE - Binary flag indicating if the TAZ is within the cordon zone
    AMZ
    LDTdistrict - 'SW', 'Central', 'NE', 'SE', 'NW'
    STATE -
    CountyName
    FIPS - FIPS code by county
    AreaType -
    Total_Employment - Total employment in by the Workplace Area Characteristics data for each census block within the TAZ boundary.
                       Obtained from LEHD 2020 wac data for the TAZ inside Ohio. For external zones, the data is obtained from the ETAZs.csv
    'Utlities_Services' - WAC data NAICS code 'CNS03':'Utilities'
    'Construction' - WAC data NAICS code CNS04 : Construction
    'Transportation_Handling' - 60% of the WAC data NAICS code CNS08: Transportation and Warehousing
    'Health_Care' - 'CNS16':'Health Care'
    'Other_Services' - 'CNS19':'Other Services
    'Government_Services' - 'CNS20':'Public Administration'
    'Highschool_Education' - 'CD01':'Less than HS Education' + 'CD02':'HS Equivalent Education'
    'College_Education' - 'CD03':'Post HS Education'
    'Hotel_Accomodation' - 'CNS18':'Accommodation and Food Services' + 'CNS11':'Real Estate and Rental and Leasing'
    'Agriculture_Production' - 'CNS01':'Agriculture, Forestry, Fishing and Hunting
    'Agriculture_Office',
    'Metal_Production' - 'CNS02':'Mining, Quarrying, and Oil and Gas Extraction'
    'Metal_Office',
    'Light_Industry_Production' -  40% of 'CNS05':'Manufacturing'
    'Light_Industry_Office',
    'Heavy_Industry_Production' - 60% of 'CNS05':'Manufacturing'
    'Heavy_Industry_Office',
    'Transportation_Equipment_Production' - 60% of 'CNS08':'Transportation and Warehousing'
    'Transportation_Equipment_Office',
    'Wholesale_Production' - 'CNS06':'Wholesale Trade'
    'Wholesale_Office',
    'Retail_Production' - 'CNS07':'Retail Trade'
    'Retail_Office',
    'CUBE_N',
    'ETAZ_NAME',
    'District',
    'CordonZone',
    'TerminalTime',
    'IMPLANTYPE',
    'IMPLANZONE',
    'FAF3_AGGR',
    'CENSUSPOP',
    'HHlt20K1to2',
    'HHlt20K3plus',
    'HH20to40K1to2',
    'HH20to40K3plus',
    'HH40to60K1to2',
    'HH40to60K3plus',
    'HH60to75K1to2',
    'HH60to75K3plus',
    'HH75to100K1to2',
    'HH75to100K3plus',
    'HH100Kplus1to2',
    'HH100Kplus3plus',
    'TotHH'


households.csv
    Household records from the synthetic population.  This is copied from
    zzSynPopH.csv, but with only the relevant fields retained.

    HHID     - Household ID
    TAZ      - TAZ ID
    PERSONS  - Number of persons in HH
    BLD      - Building Size (see ACS PUMS documentation)
    INCOME   - Annual household income

persons.csv
    Person records from the synthetic population.  This is copied from
    zzSynPopP.csv, but with the names slightly modified for ActivitySim.
    See PUMS documentation for definition of codes/categories.

    HHID            Household ID
    PERSID          Person ID
    SEX             Sex (1=Male, 2=Female)
    AGE             Age
    SCH             School Enrollment (b=NA for <3 yrs old, 1=No, 2=Yes Public, 3=Yes Private)
    ESR             Employment Status (b=NA for<16 yrs old, 1,2=Yes Civilian, 3=Unemployed, 4,5=Yes Armed forces, 6=Not in labor force)
    NAICSP02        NAICS Code for industry
    SOCP10          SOCP Standard Occupantional Classification
    SW_UNSPLIT_IND  Ohio SW Model unsplit industry
    SW_OCCUP        Ohio SW Model occupation
    SW_SPLIT_IND    Ohio SW Model split industry

skims.omx
    Skim matrices converted from TP+ to OMX format.
