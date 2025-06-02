let 
    fxFunction =
        (
            tableToTransform as table,
            modelTable as table 
        ) as table =>
            let
                _model_table_names = // Get column names of model table.
                    Table.ColumnNames ( modelTable ),
                _model_table_types_text = // Get names of column types of model table.
                    Table.Schema ( modelTable )[Kind],
                _get_model_table_types_from_text = // For each column type name, get column type.
                    List.Transform (
                        _model_table_types_text,
                        each
                            Record.Field (
                                [
                                    any = type any,
                                    binary = type binary,
                                    date = type date,
                                    datetime = type datetime,
                                    datetimezone = type datetimezone,
                                    duration = type duration,
                                    list = type list,
                                    logical = type logical,
                                    null = type any,
                                    number = type number,
                                    record = type record,
                                    table = type table,
                                    text = type text,
                                    time = type time
                                ],
                                _
                            )
                    ),
                _combine_names_and_types = // Create list of column names and types.
                    List.Zip (
                        {
                            _model_table_names,
                            _get_model_table_types_from_text
                        }
                    ),
                _change_types = // Apply each column name and type to tableToTransform
                    List.Accumulate (
                        _combine_names_and_types,
                        tableToTransform,
                        ( state, current ) => 
                            Table.TransformColumnTypes (
                                state,
                                {
                                    {
                                        current{0},
                                        current{1}
                                    }
                                }
                            )
                    )
            in
                _change_types,
    fxDocumentation =
        type function (
            tableToTransform as (
                type table meta [
                    Documentation.FieldCaption = "Table to Transform",
                    Documentation.FieldDescription = "The table on which you want to change the column types."
                ]
            ),
            modelTable as (
                type table meta [
                    Documentation.FieldCaption = "Model Table",
                    Documentation.FieldDescription = "The table containing the model column types."
                ]
            )            
        ) as table meta
            [
                Documentation.Name = "fxSetColumnTypesFromModel",
                Documentation.Description = "This function replaces the column types of one table with the column types of another table with the same column names.",
                Documentation.Category = "Clean"
            ],
    fxReplaceMeta =
        Value.ReplaceType (
            fxFunction,
            fxDocumentation
        ) 
in
    fxReplaceMeta
