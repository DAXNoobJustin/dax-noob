let 
    fxFunction =
        (
            tableToTransform as table,
            columnsToTransform as list,
            optional replaceNumbers as logical // if set to false, columns with number type are ignored
        ) as table =>
            let 
                _table_schema =
                    Table.Buffer (
                        Table.SelectRows (
                            Table.Schema ( tableToTransform ),
                            each
                                List.Contains (
                                    columnsToTransform,
                                    [Name]
                                )
                        )
                    ),  
                _replace_skey_null =
                    Table.ReplaceValue (
                        tableToTransform,
                        null,
                        -1,
                        Replacer.ReplaceValue,
                        Table.SelectRows (
                            _table_schema,
                            each
                                Text.Start (
                                    [Name],
                                    6
                                ) = "_SKey " 
                        )[Name]
                    ),                    
                _replace_text_null =
                    Table.TransformColumns (
                        _replace_skey_null,
                        List.Transform (
                            Table.SelectRows (
                                _table_schema,
                                each
                                    [Kind] = "text"
                            )[Name],
                            each {
                                _,
                                each 
                                    if
                                        List.Contains (
                                            {
                                                null,
                                                "blank",
                                                "empty",
                                                "unknown",
                                                "null",                                  
                                                ""
                                            },   
                                            Text.Lower ( _ )                             
                                        )
                                    then 
                                        "Unknown"
                                    else
                                        _,
                                type
                                    text
                            }
                        )
                    ),  
                _replace_number_null =
                    if 
                        replaceNumbers ?? false
                    then
                        Table.ReplaceValue (
                            _replace_text_null,
                            null,
                            0,
                            Replacer.ReplaceValue,
                            Table.SelectRows (
                                _table_schema,
                                each
                                    [Kind] = "number"
                            )[Name]
                        )     
                    else
                        _replace_text_null
            in
                _replace_number_null,
    fxDocumentation =
        type function (
            tableToTransform as (
                type table meta [
                    Documentation.FieldCaption = "Table to Transform",
                    Documentation.FieldDescription = "The table you want to replace values on."
                ]
            ),
            columnsToTransform as (
                type list meta [
                    Documentation.FieldCaption = "Columns To Transform",
                    Documentation.FieldDescription = "The list of column names to perform the transformation on."
                ]
            ),
            optional replaceNumbers as (
                type logical meta [
                    Documentation.FieldCaption = "Replace Numbers",
                    Documentation.FieldDescription = "Boolean flag to determine if null numbers should be replaced with 0."
                ]
            )                           
        ) as table meta
            [
                Documentation.Name = "fxReplaceWithDefaultValue",
                Documentation.Description = "Replaces nulls and like null values with a default, standard value.",
                Documentation.Category = "Clean",
                Documentation.Examples =
                    {
                        [
                            Description = "",
                            Code =
                                "
fxReplaceWithDefaultValues ( #table (), true, true )
TextCol, NumberCol
Hello, 1
Test, 2,
Justin, null,
(blank), 3,
BLANK, 4
                                ",
                            Result =
                                "
TextCol, NumberCol
Hello, 1
Test, 2,
Justin, 0,
Unknown, 3,
Unknown, 4
                                "                         
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