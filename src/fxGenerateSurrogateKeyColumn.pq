let
    fxFunction =
        (
            tableToTransform as table,
            newSKeyColumnName as text,
            optional columnsToGroupBy as list  
        ) as table =>
            let
                _cleaned_skey_column = // Set new surrogate key column name.
                    "_SKey " & newSKeyColumnName,
                _create_table_group = 
                    if
                        columnsToGroupBy is null
                    then // If null, a unique surrogate key is generated for each record in the table.  
                        Table.AddIndexColumn (
                            tableToTransform,
                            _cleaned_skey_column,
                            0,
                            1,
                            Int64.Type
                        )                
                    else // If not null, surrogate key is generated for each grouping of provided columns.
                        let
                            _create_distinct_table_group = // Create a table of distinct value combinations for the columns provided and then add index. 
                                Table.AddIndexColumn (
                                    Table.Distinct (
                                        Table.SelectColumns (
                                            tableToTransform,
                                            columnsToGroupBy
                                        )
                                    ),
                                    _cleaned_skey_column,
                                    0,
                                    1,
                                    Int64.Type
                                ),
                            _join_and_expand_table = // Join generated surrogate key back to original table. 
                                fxJoinAndExpandTable (
                                    1,
                                    tableToTransform,
                                    _create_distinct_table_group,
                                    columnsToGroupBy,
                                    { _cleaned_skey_column }
                                )
                        in
                            _join_and_expand_table               
            in
                _create_table_group,
    fxDocumentation = 
        type function (
            tableToTransform as (
                type table meta [
                    Documentation.FieldCaption = "Table to Transform",
                    Documentation.FieldDescription = "The table on which to generate a new surrogate key column."
                ]
            ),
            newSKeyColumnName as (
                type text meta [
                    Documentation.FieldCaption = "New SKey Column Name",
                    Documentation.FieldDescription = "The new column name for the surrogate key. The input will be prefixed with ""_SKey"".",
                    Documentation.SampleValues = 
                        {
                            "Geo",
                            "Parcel",
                            "Ticket"
                        }
                ]
            ),
            optional columnsToGroupBy as (
                type list meta [
                    Documentation.FieldCaption = "Columns To Group By",
                    Documentation.FieldDescription = "The list of columns used to generate the surrogate key. A unique key will be generated for each unique combination of values. If left blank, each row will be considered unique.",
                    Documentation.SampleValues = 
                        {
                            "null",
                            "{ ""AZ"", ""Metro"" }"
                        }
                ]
            )                        
        ) as table meta
            [
                Documentation.Name = "fxGenerateSurrogateKeyColumn",
                Documentation.Description = "This function takes a table, a new SKey column name, and an optional list of columns to group by, and returns a new table with an added SKey column that assigns a unique index to each row or group of rows.",
                Documentation.Category = "Generate",
                Documentation.Examples =
                    {
                        [
                            Description = "",
                            Code = "fxGenerateSurrogateKeyColumn ( dimParcel, ""Parcel"", {""Parcel Id""} )",
                            Result = "dimParcel table with new surrogate key column called ""_SKey Parcel""."
                        ]
                    }
            ],
    fxReplaceMeta =
        Value.ReplaceType (
            fxFunction,
            fxDocumentation
        ) 
in
    fxReplaceMeta