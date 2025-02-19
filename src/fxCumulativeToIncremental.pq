let
    fxFunction =
        (
            tableToTransform as table,
            groupByColumns as list,
            sortByColumn as text,
            columnsToTransform as list
        ) as table =>
            let
                _create_groups = // Group all rows by group by columns
                    Table.Group (
                        tableToTransform,
                        groupByColumns,
                        {
                            {
                                "Group",
                                each
                                    _,
                                type
                                    table     
                            }
                        }
                    ),
                _create_incremental_data = // For each set of grouped rows, create incremental column
                    Table.AddColumn (
                        _create_groups,
                        "Data",
                        each
                            let
                                _sort_by_date = // Sort data by sort by column
                                    Table.Sort (
                                        [Group],
                                        { { sortByColumn, Order.Ascending } }
                                    ),
                                _create_index = // Add index to be used for calculating difference between each cumulative row
                                    Table.AddIndexColumn (
                                        _sort_by_date,
                                        "Index",
                                        0,
                                        1,
                                        Int64.Type
                                    ),
                                _create_incremental_data_for_each_column = // For each cumulative column, create incremental column
                                    List.Accumulate(
                                        columnsToTransform,
                                        _create_index,
                                        (
                                            state,
                                            columnToTransform
                                        ) =>
                                            let 
                                                _buffer_data = // Buffer results
                                                    List.Buffer (
                                                        Table.Column(state, columnToTransform)
                                                    ),
                                                _calculate_incremental_value = // Add column with calculated incremental value
                                                    Table.AddColumn (
                                                        state,
                                                        columnToTransform & " Incremental",
                                                        each
                                                            if
                                                                [Index] > 0
                                                            then                              
                                                                Record.Field(_, columnToTransform) -
                                                                    _buffer_data { [Index] - 1 }
                                                            else 
                                                                Record.Field(_, columnToTransform),
                                                        type
                                                            number
                                                    )  
                                                in
                                                    _calculate_incremental_value           
                                    )
                            in
                                _create_incremental_data_for_each_column
                    ),
                _columns_to_extract = // Get list of columns to extract from group
                    List.Combine ( 
                        {
                            List.RemoveItems (
                                Table.ColumnNames ( tableToTransform ),
                                groupByColumns
                            ),
                            List.Transform (
                                columnsToTransform,
                                each
                                    _ & " Incremental"
                            )
                        }
                    ),
                _extract_incremental_data = // Expand columns from group
                    Table.ExpandTableColumn (
                        _create_incremental_data,
                        "Data",
                        _columns_to_extract,
                        _columns_to_extract
                    ),
                _remove_group_column = // Remove group column
                    Table.RemoveColumns (
                        _extract_incremental_data,
                        { "Group" }
                    ),
                _restore_column_types = // Set column types
                    fxSetColumnTypesFromModel (
                        _remove_group_column,
                        List.Accumulate(
                            columnsToTransform,
                            tableToTransform,
                            (
                                state,
                                column
                            ) =>                        
                                Table.AddColumn (
                                    state,
                                    column & " Incremental",
                                    each
                                        null,
                                    type
                                        number
                                )
                        )
                    )                  
            in
                _restore_column_types,
    fxDocumentation =
        type function (
            tableToTransform as (
                type table meta [
                    Documentation.FieldCaption = "Table to Transform",
                    Documentation.FieldDescription = "The table containing the cumulative columns to be transformed."
                ]
            ),    
            groupByColumns as (
                type list meta [
                    Documentation.FieldCaption = "Group By Columns",
                    Documentation.FieldDescription = "The columns that, when taken together, the running total is applied to."
                ]
            ),
            sortByColumn as (
                type text meta [
                    Documentation.FieldCaption = "Sort by Column",
                    Documentation.FieldDescription = "The time/sort by column for the running total."
                ]
            ),            
            columnsToTransform as (
                type list meta [
                    Documentation.FieldCaption = "Columns to Transform",
                    Documentation.FieldDescription = "The cumulative columns to be transformed to incremental."
                ]
            )                                                                   
        ) as table meta
            [
                Documentation.Name = "fxCumulativeToIncremental",
                Documentation.Description = "This function transforms one or more running total columns into incremental columns.",
                Documentation.Category = "Generate"
            ],
    fxReplaceMeta =
        Value.ReplaceType (
            fxFunction,
            fxDocumentation
        ) 
in
    fxReplaceMeta