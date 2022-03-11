from cfgy import *

@configclass
class InputTable:
    """
    Definition of a single input table to be read by ActivitySim.
    """
    tablename: str = RequireString(default=NODEFAULT, doc="""
    Name of table
    
    This should generally be one of ActivitySim canonical names (households, 
    persons, land_use, etc).  This name is used by ORCA to store and retrieve
    the table, so using an unexpected name will prevent most components from
    accessing the data.
    """)
    filename: str = RequireString(doc="""
    Relative path within Input directory.
    """)
    index_col: str = RequireString(doc="""
    The column that should be treated as the index.
    """)
    rename_columns: dict = RequireDictOfStrTo(str, doc="""
    Mapping of existing column names to used column names.
    
    Many ActivitySim components expect particular data to be stored in
    particular named columns. If the input data file on disk has names 
    that are not consistent with the expectations, this setting can be
    used to remap columns at read time. 
    """)
    keep_columns: list = RequireSetOf(str, doc="""
    Set of column names to load.
    
    Columns in the data file other than these will be ignored.  Note that these
    names are the names to keep *after* the `rename_columns` transformation. 
    """)


@configclass
class Settings:
    """
    Top level settings for ActivitySim
    """

    input_table_list: list = RequireListOf(InputTable, doc="""
    List of settings for each input table.
    """)

    create_input_store: bool = RequireBool(doc="""
    Convert input CSVs to HDF5 format and save them to outputs directory
    
    A new `input_data.h5` file will be created in the outputs folder using 
    data from CSVs from `input_table_list` to use for subsequent model runs.
    """)

    resume_after: str = RequireString(doc="""
    Resume execution after this component.
    
    There are two ways to use this setting:
    
    - Set to the name of a model component, to resume from after that 
      component.  This presumes that a suitable checkpoint is available.
    
    - Set this to a single underscore '_' to resume after the last 
      successful checkpoint of any component.    
    """)

    models: list = RequireListOf(str, doc="""
    list of model steps to run - auto ownership, tour frequency, etc.
    """)

    households_sample_size: int = RequireInteger(doc="""
    Number of households to sample and simulate.
    
    Set to zero or omit to simulate all households.
    """)

    check_for_variability: bool = RequireBool(default=False, doc="""
    Check for variability in an expression result 
    
    This is a debugging feature, leave it off to speed-up runtime
    """)

    use_shadow_pricing: bool = RequireBool(default=False, doc="""
    turn shadow_pricing on and off for work and school location
    """)

    chunk_size: int = RequireInteger(doc="""
    Approximate amount of RAM in bytes to allocate to ActivitySim for batch processing.
     
    For more info, see :ref:`chunk_size`.
    """)

