let
    fxFunction =
        ( 
            rowIntegritySetting as number,          
            sourceTable as table,
            targetTable as table,
            sourceJoinColumns as list,
            columnsToExpand as list,
            optional targetJoinColumns as list,            
            optional alternateColumnsToExpand as list
        ) as table =>
            let 
                _temp_column_name = 
                    "_TempJoin",
                _columns_to_join_clean = // use if target join columns different than source join columns
                    targetJoinColumns ?? sourceJoinColumns,
                _columns_to_expand_clean = // use if renaming columns to expand
                    alternateColumnsToExpand ?? columnsToExpand,
                _target_table_clean = // force row integrity on target table
                    if 
                        rowIntegritySetting = 2
                    then
                        Table.Distinct (
                            targetTable,
                            _columns_to_join_clean
                        )
                    else
                        targetTable,
                _join_tables = // perform join
                    Table.NestedJoin (
                        sourceTable,
                        sourceJoinColumns,
                        _target_table_clean,
                        _columns_to_join_clean,
                        _temp_column_name,
                        JoinKind.LeftOuter
                    ),
                _expand_join = // expand columns
                    Table.ExpandTableColumn (
                        _join_tables,
                        _temp_column_name,
                        columnsToExpand,
                        _columns_to_expand_clean
                    ),
                _check_row_integrity = // check if the row counts on the source table are the same after joining the target table
                    if
                        rowIntegritySetting = 1 and
                            Table.RowCount ( sourceTable ) <> Table.RowCount ( _expand_join )
                    then
                        error "Left join has generated additional rows" 
                    else
                        _expand_join
            in 
                _check_row_integrity,
    fxDocumentation =
        type function (
            rowIntegritySetting as (
                type number meta [
                    Documentation.FieldCaption = "Row Integrity Setting",
                    fxDocumentation.AllowedValues = { 0, 1, 2 },
                    Documentation.FieldDescription =
"0.  None - Do nothing.
1. Check Row Integrity - Check to see if the beginning and end row counts are the same in the source table after joining the target table. If not, throw an error.
2. Force Row Integrity - Force a Table.Distinct on the join columns for the target table."
                ]
            ),    
            sourceTable as (
                type table meta [
                    Documentation.FieldCaption = "Source Table",
                    Documentation.FieldDescription = "The source table to be joined to."
                ]
            ),
            targetTable as (
                type table meta [
                    Documentation.FieldCaption = "Target Table",
                    Documentation.FieldDescription = "The target table to join."
                ]
            ),            
            sourceJoinColumns as (
                type list meta [
                    Documentation.FieldCaption = "Source Join Columns",
                    Documentation.FieldDescription = "The list of column names to join. Must be the same in both tables if targetJoinColumns is null."
                ]
            ),
            columnsToExpand as (
                type list meta [
                    Documentation.FieldCaption = "Columns To Expand",
                    Documentation.FieldDescription = "The list of column names you want to expand from the target table."
                ]
            ),
            optional targetJoinColumns as (
                type list meta [
                    Documentation.FieldCaption = "Target Join Columns",
                    Documentation.FieldDescription = "Optional list of column names in the target table to join. Leave null if the column names are the same in the source table."
                ]
            ),            
            optional alternateColumnsToExpand as (
                type list meta [
                    Documentation.FieldCaption = "Alternate Columns To Expand",
                    Documentation.FieldDescription = "Optional list of column names to expand. Columns expanded in columnsToExpand will be renamed to this list when joined."
                ]
            )                                                                      
        ) as table meta
            [
                Documentation.Name = "fxJoinAndExpandTable",
                Documentation.Description = "This function performs a left join on a source and target table and expands the list of provided columns.",
                Documentation.Category = "Generate"
            ],
    fxReplaceMeta =
        Value.ReplaceType (
            fxFunction,
            fxDocumentation
        ) 
in
    fxReplaceMeta
