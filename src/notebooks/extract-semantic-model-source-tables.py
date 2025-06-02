import re
import pandas as pd
import sempy.fabric as fabric
from typing import List, Optional
 
workspaces = [
    'Workspace 1',
    'Workspace 2',
]
m_expression_patterns = [
    r'',
]
 
def get_table_sources_from_workspaces(
    workspaces: List[str],
    expr_patterns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Retrieve source tables for all Fabric semantic models in the given workspaces.
 
    :param workspaces: List of workspace names.
    :param expr_patterns: Optional list of regex strings to extract names from M expressions.
    :return: pandas DataFrame with columns: workspace, model_name, table_name, source_table_name.
    """
    patterns = [re.compile(p) for p in (expr_patterns or [])]
    records = []
 
    for ws in workspaces:
        server = fabric.create_tom_server(readonly=True, workspace=ws)
        dataset_names = fabric.list_datasets(workspace=ws)['Dataset Name']
 
        for model_name in dataset_names:
            model = server.Databases.GetByName(model_name).Model
 
            for table in model.Tables:
                for part in table.Partitions:
                    src = part.Source
                    source = getattr(src, 'EntityName', None)
                    if not source:
                        expr = getattr(src, 'Expression', '') or ''
                        for pat in patterns:
                            m = pat.search(expr)
                            if m:
                                source = m.group(1)
                                break
                        source = source or expr or None
 
                    records.append({
                        'workspace': ws,
                        'model_name': model_name,
                        'table_name': table.Name,
                        'source_table_name': source,
                    })
 
    df = pd.DataFrame(records)
    return df.sort_values(['workspace', 'model_name', 'table_name']).reset_index(drop=True)
 
df = get_table_sources_from_workspaces(workspaces, m_expression_patterns)
display(df)
