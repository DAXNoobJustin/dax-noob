let
    fxFunction =
        (
            textToTransform as text
        ) as text =>
            let
                fxIsUpper =
                    ( char as text ) as logical =>
                        Text.Upper ( char ) = char,            
                _reverse_text =
                    Text.Reverse ( textToTransform ),
                _list_of_characters =
                    Text.ToList ( _reverse_text ),
                _create_reversed_text =
                    List.Accumulate (
                        _list_of_characters,
                        [ 
                            is_upper_previous = true,
                            transformed_text = ""
                        ],
                        ( state, current ) => 
                            [
                                is_upper_previous =
                                    if
                                        current = " "
                                    then 
                                        true
                                    else
                                        fxIsUpper ( current ),
                                transformed_text =                   
                                    if
                                        current = " "
                                    then
                                        state[transformed_text]
                                    else if 
                                        Text.End ( state[transformed_text], 1 ) = " "
                                    then 
                                        state[transformed_text] & current
                                    else if 
                                        fxIsUpper ( current ) and 
                                            not state[is_upper_previous]
                                    then
                                        state[transformed_text] & current & " "
                                    else if  
                                        not fxIsUpper ( current ) and 
                                            state[is_upper_previous]
                                    then
                                        state[transformed_text] & " " & current
                                    else                        
                                        state[transformed_text] & current
                            ]
                    )[transformed_text],
                _restore_text_order =
                    Text.Reverse ( _create_reversed_text ),
                _trim_text =
                    Text.Trim ( _restore_text_order ),
                _capitalize_first_word =
                    Text.Upper ( Text.Start ( _trim_text, 1 ) ) &
                        Text.End ( _trim_text, Text.Length ( _trim_text ) - 1 )     
            in
                _capitalize_first_word,
    fxDocumentation =
        type function (
            textToTransform as (
                type text meta [
                    Documentation.FieldCaption = "Text to Transform",
                    Documentation.FieldDescription = "The text stored in camel or pascal case.",
                    Documentation.SampleValues = 
                        {
                            "helloWorld",
                            "ThisIsASampleText"
                        }
                ]
            )                     
        ) as text meta
            [
                Documentation.Name = "fxSplitCamelCaseText",
                Documentation.Description = "This function splits a string stored in camel or pascal case into multiple words separated by spaces.",
                Documentation.Category = "Clean",
                Documentation.Examples =
                    {
                        [
                            Description = "",
                            Code = "fxSplitCamelCaseText ( ""helloWorld"" )",
                            Result = "Hello World"
                        ],
                        [
                            Description = "",
                            Code = "fxSplitCamelCaseText ( ""ThisIsAnExampleString"" )",
                            Result = "This Is An Example String"
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