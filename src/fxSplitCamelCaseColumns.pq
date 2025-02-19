let
    fxFunction =
        (
            tableToTransform as table,
            optional columnNameOverrides as nullable list
        ) as table =>
            let
                _source_column_names =
                    Table.ColumnNames ( tableToTransform ),
                _clean_column_name_overrides =
                    columnNameOverrides ?? {},            
                _columns_to_override =
                    List.Transform (
                        _clean_column_name_overrides,
                        each _{ 0 }
                    ),                
                _columns_to_transform =
                    List.RemoveItems (
                        _source_column_names,
                        _columns_to_override
                    ),            
                _transform_source_column_names =
                    List.Transform (
                        _columns_to_transform,
                        each
                            {
                                _,
                                fxSplitCamelCaseText ( _ )
                            }
                    ),
                _combine_lists =
                    List.Combine (
                        {
                            _clean_column_name_overrides,
                            _transform_source_column_names
                        }
                    ),
                _apply_changes =
                    Table.RenameColumns (
                        tableToTransform,
                        _combine_lists
                    )
            in
                _apply_changes,
    fxDocumentation =
        type function (
            tableToTransform as (
                type table meta [
                    Documentation.FieldCaption = "Table To Transform",
                    Documentation.FieldDescription = "The table whose column names you want to transform.",
                    Documentation.SampleValues = 
                        {
                            "#table()"
                        }
                ]
            ),
            optional columnNameOverrides as (
                type nullable list meta [
                    Documentation.FieldCaption = "Column Name Overrides",
                    Documentation.FieldDescription = "A list of column name overrides to be used instead of the default transformation.",
                    Documentation.SampleValues = 
                        {
                            "{ ""OldColumnName"", ""Override Column Name"", }"
                        }
                ]
            )                                 
        ) as text meta
            [
                Documentation.Name = "fxSplitCamelCaseColumns",
                Documentation.Description = "This function splits all camel case column names in a table and converts them to proper case. The function also takes an optional list of alternative column name transformations.",
                Documentation.Category = "Clean"
            ],
    fxReplaceMeta =
        Value.ReplaceType (
            fxFunction,
            fxDocumentation
        ) 
in
    fxReplaceMeta