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
    :return: pandas DataFrame with columns:
        - workspace: Name of the workspace.
        - model_name: Name of the semantic model (dataset).
        - table_name: Name of the table in the model.
        - source_table_name: Name of the source table or expression.
        - data_source_name: Display name of the source item (Lakehouse or Warehouse name).
        - data_source_type: Type of the SQL endpoint (e.g., Lakehouse(SQLEndpoint is displayed since it use lakehouse's SQL endpoint), Warehouse).
        - sql_endpoint_connection_string: Connection string for the SQL endpoint.
    """
    patterns = [re.compile(p) for p in (expr_patterns or [])]
    records = []

    # Build workspace name→ID map once
    df_ws = fabric.list_workspaces().rename(columns={'Name':'workspace','Id':'workspace_id'})
    ws_map = dict(zip(df_ws['workspace'], df_ws['workspace_id']))

    # Single REST client, to retrieve endpoints
    client = fabric.FabricRestClient()

    for ws in workspaces:
        # force TOM metadata to be refreshed (for new datasets)
        fabric.refresh_tom_cache(ws)
        server = fabric.create_tom_server(readonly=True, workspace=ws)
        dataset_names = fabric.list_datasets(workspace=ws)['Dataset Name']

        # Lookup correct workspace_id
        ws_id = ws_map.get(ws)

        # preload all lakehouse & warehouse endpoints
        lake_map, wh_map = {}, {}
        for kind in ('lakehouses','warehouses'):
            resp = client.get(f"/v1/workspaces/{ws_id}/{kind}")
            if resp.status_code == 200:
                for item in resp.json().get('value', []):
                    props    = item.get('properties', {}) or {}
                    ep_props = props.get('sqlEndpointProperties') or {}
                    ep_id    = ep_props.get('id') or item.get('id')
                    conn     = ep_props.get('connectionString') or props.get('connectionString')
                    if ep_id and conn:
                        (lake_map if kind=='lakehouses' else wh_map)[ep_id] = conn

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

                    # initialize extra metadata fields
                    data_source_name               = None
                    data_source_type               = None
                    sql_endpoint_connection_string = None

                    df_expr = fabric.list_expressions(dataset=model_name, workspace=ws)
                    dbq = df_expr[df_expr['Name']=='DatabaseQuery']
                    if not dbq.empty:
                        matches = re.findall(r'"([^"]*)"', dbq['Expression'].iloc[0])
                        if len(matches) > 1:
                            sql_ep_id = matches[1]
                            # lookup name & type from items
                            items = fabric.list_items(workspace=ws)
                            itm = items[items['Id']==sql_ep_id]
                            if not itm.empty:
                                data_source_name = itm['Display Name'].iloc[0]
                                data_source_type = itm['Type'].iloc[0]
                            # Lookup connectionString from preloaded maps
                            sql_endpoint_connection_string = (
                                lake_map.get(sql_ep_id)
                                or wh_map.get(sql_ep_id)
                            )

                    records.append({
                        'workspace': ws,
                        'model_name': model_name,
                        'table_name': table.Name,
                        'source_table_name': source,
                        #  Include the new columns
                        'data_source_name': data_source_name,
                        'data_source_type': data_source_type,
                        'sql_endpoint_connection_string': sql_endpoint_connection_string
                    })

    df = pd.DataFrame(records)
    return df.sort_values(['workspace', 'model_name', 'table_name']).reset_index(drop=True)
 
df = get_table_sources_from_workspaces(workspaces, m_expression_patterns)
display(df)
