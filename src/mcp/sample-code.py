#!/usr/bin/env python
# coding: utf-8

# ## DAXOptimizer - V2
# 
# New notebook

# In[ ]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install semantic-link-labs -q
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install -q --force-reinstall openai==1.30 httpx==0.27.0
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install -q --force-reinstall https://mmlspark.blob.core.windows.net/pip/1.0.11-spark3.5/synapseml_core-1.0.11.dev1-py2.py3-none-any.whl
# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install -q --force-reinstall https://mmlspark.blob.core.windows.net/pip/1.0.11.1-spark3.5/synapseml_internal-1.0.11.1.dev1-py2.py3-none-any.whl


# In[2]:


import builtins
import functools
import io
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from contextlib import contextmanager
from threading import local
from typing import Any, Callable, Generator, Type
from uuid import uuid4


import pandas as pd
import sempy.fabric as fabric
import sempy_labs as labs
import synapse.ml.aifunc as aifunc
from synapse.ml.aifunc import Conf


# In[3]:


# Spark settings
spark.conf.set("spark.sql.caseSensitive", True)

# -- Semantic Model connection --
model = {
    "name": "MSBilledPipelineCurated_VOrder_Segmented",
    "workspace_name": "MSXI-BilledPipeline-Sandbox",
}

# -- Lakehouse for logs & resources --
log_table_name = "testing_logs"
lakehouse_mount_path = "/default"

# -- Folder where user‐uploaded resources reside --
resource_folder_path = "/Files/ResourceDocs"

# -- Optimization parameters --
run_id = str(uuid4())
ai_model = "gpt-4o"
ai_temperature = 0.5

# -- Target DAX query (string) --
target_dax_query = """
// DAX Query
DEFINE
    VAR __DS0FilterTable =
        FILTER (
            KEEPFILTERS (
                SUMMARIZE (
                    VALUES ( 'Product' ),
                    'Product'[Pipeline Grouping 1],
                    'Product'[Pipeline Grouping 2]
                )
            ),
            AND (
                'Product'[Pipeline Grouping 1]
                    IN { "Commercial excl. Azure", "Unassigned", BLANK () },
                NOT (
                ( 'Product'[Pipeline Grouping 1], 'Product'[Pipeline Grouping 2] )
                    IN { ( "Commercial excl. Azure", "ISD" ) } )
            )
        )
    VAR __DS0FilterTable2 =
        TREATAS ( { 1 }, 'Account Team'[ATU Flag] )
    VAR __DS0FilterTable3 =
        TREATAS ( { "Deal Based Pipeline" }, 'Pipeline'[Pipeline Category] )
    VAR __DS0FilterTable4 =
        FILTER (
            KEEPFILTERS ( VALUES ( 'Account'[TPID] ) ),
            NOT ( ISBLANK ( 'Account'[TPID] ) )
        )
    VAR __DS0FilterTable5 =
        TREATAS ( { "Yes" }, 'Dynamic Filters'[Billed Annualized Switch Flag] )
    VAR __DS0FilterTable6 =
        TREATAS (
            { "Field", "Stores Field", "Unassigned", "Services" },
            'Business'[Business Summary]
        )
    VAR __DS0FilterTable7 =
        TREATAS (
            { "Enterprise Commercial", "Enterprise Public Sector", "SM&C Corporate" },
            'Segment'[Field Summary Segment]
        )
    VAR __DS0FilterTable8 =
        FILTER (
            KEEPFILTERS (
                SUMMARIZE (
                    VALUES ( 'Calendar' ),
                    'Calendar'[Fiscal Year],
                    'Calendar'[Relative Quarter]
                )
            ),
            AND (
                'Calendar'[Fiscal Year] IN { "FY25", "FY26" },
                ( 'Calendar'[Fiscal Year], 'Calendar'[Relative Quarter] )
                    IN { ( "FY25", "CQ" ) }
            )
        )
    VAR __DS0FilterTable9 =
        FILTER (
            KEEPFILTERS ( VALUES ( 'Sales Program'[Sales Program] ) ),
            NOT ( 'Sales Program'[Sales Program]
                IN { "SecNumCloud", "Sovereign Cloud | SecNumCloud" } )
        )
    VAR __DS0Core =
        SUMMARIZECOLUMNS (
            ROLLUPADDISSUBTOTAL (
                'Account'[Time Zone],
                "IsGrandTotalRowTotal",
                'Account'[Field Area],
                "IsDM1Total",
                'Account'[Field Accountability Unit],
                "IsDM3Total"
            ),
            __DS0FilterTable,
            __DS0FilterTable2,
            __DS0FilterTable3,
            __DS0FilterTable4,
            __DS0FilterTable5,
            __DS0FilterTable6,
            __DS0FilterTable7,
            __DS0FilterTable8,
            __DS0FilterTable9,
            "Qualified_Pipeline", 'Pipeline'[Qualified Pipeline],
            "QPC_to_Target__", 'Target'[QPC to Target %],
            "CPC_to_Target__", 'Target'[CPC to Target %],
            "QPC_Goal", 'Close Rate'[QPC Goal],
            "Actual_Revenue", 'Revenue'[Actual Revenue],
            "Scheduled_Revenue", 'Revenue'[Scheduled Revenue],
            "Net_New_Revenue_Required", 'Target'[Net New Revenue Required],
            "Committed_Pipeline", 'Pipeline'[Committed Pipeline],
            "Committed_at_Risk_Pipeline", 'Pipeline'[Committed at Risk Pipeline],
            "Run_Rate_Projections2", 'Run Rate Projections'[Run Rate Projections],
            "Total_DBF", 'Revenue'[Total DBF],
            "DBF_vs_Target", 'Target'[DBF vs Target],
            "Uncommitted_Pipeline", 'Pipeline'[Uncommitted Pipeline],
            "DBF_YoY__", 'Revenue'[DBF YoY $],
            "DBF__YoY__", 'Revenue'[DBF  YoY %],
            "Cloud_Mix__", 'Pipeline'[Cloud Mix %],
            "Target2", 'Target'[Target],
            "DBF_VTT__", 'Target'[DBF VTT %],
            "Upside_Pipeline", 'Pipeline'[Upside Pipeline],
            "Qualified_Pipeline_in_High_Sales_Stage", 'Pipeline'[Qualified Pipeline in High Sales Stage],
            "Unqualified_Pipeline", 'Pipeline'[Unqualified Pipeline],
            "v_QPC_to_Target___Status", IGNORE ( 'Target'[_QPC to Target % Status] )
        )

EVALUATE
__DS0Core

"""

additional_context = ""


# In[4]:


# Define the expected schema for DAX trace log events.
# Field names should match exactly what the trace returns.
event_schema = {
    "DirectQueryBegin": [
        "EventClass", "CurrentTime", "TextData", "StartTime", 
        "EndTime", "Duration", "CpuTime", "Success", "SessionID"
    ],
    "DirectQueryEnd": [
        "EventClass", "CurrentTime", "TextData", "StartTime", 
        "EndTime", "Duration", "CpuTime", "Success", "SessionID"
    ],
    "VertiPaqSEQueryBegin": [
        "EventClass", "EventSubclass", "CurrentTime", 
        "TextData", "StartTime", "SessionID"
    ],
    "VertiPaqSEQueryEnd": [
        "EventClass", "EventSubclass", "CurrentTime", "TextData", 
        "StartTime", "EndTime", "Duration", "CpuTime", "Success", "SessionID"
    ],
    "VertiPaqSEQueryCacheMatch": [
        "EventClass", "EventSubclass", "CurrentTime", "TextData", "SessionID"
    ],
    "QueryBegin": [
        "EventClass", "EventSubclass", "CurrentTime", "TextData", 
        "StartTime", "ConnectionID", "SessionID", "RequestProperties"
    ],
    "QueryEnd": [
        "EventClass", "EventSubclass", "CurrentTime", "TextData", 
        "StartTime", "EndTime", "Duration", "CpuTime", "Success", 
        "ConnectionID", "SessionID"
    ],
}


# In[5]:


# Thread‐local storage variable to manage indentation in logs.
_thread_local = local()

@contextmanager
def dynamic_indented_print() -> Generator[None, None, None]:
    """
    Temporarily overrides built‐in print to indent messages based on call depth.
    """
    original_print = builtins.print

    def custom_print(*args: Any, **kwargs: Any) -> None:
        depth = getattr(_thread_local, "call_depth", 0)
        indent = "    " * depth
        original_print(indent + " ".join(map(str, args)), **kwargs)

    builtins.print = custom_print
    try:
        yield
    finally:
        builtins.print = original_print

def log_function_calls(func: Callable) -> Callable:
    """
    Decorator that logs the start and end of a function call with indentation.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not hasattr(_thread_local, "call_depth"):
            _thread_local.call_depth = 0

        with dynamic_indented_print():
            print(f"✅ {func.__name__} - Starting")
            _thread_local.call_depth += 1
            try:
                result = func(*args, **kwargs)
            finally:
                _thread_local.call_depth -= 1
                print(f"✅ {func.__name__} - Ending")
        return result

    return wrapper

def retry(exceptions: tuple[Type[Exception], ...],
          tries: int = 3,
          delay: int = 5,
          backoff: int = 2,
          logger: Callable = print) -> Callable:
    """
    Decorator for retrying a function call with exponential backoff.
    """
    def decorator_retry(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_retry(*args: Any, **kwargs: Any) -> Any:
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # with dynamic_indented_print():
                    #     logger(f"⚠️ {func.__name__} failed with {e}, retrying in {_delay} seconds...")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry


# In[6]:


def trace_started(traces: pd.DataFrame, trace_name: str) -> bool:
    """
    Checks if a specific trace (by name) has started by inspecting the traces DataFrame.
    """
    return traces.loc[traces["Name"] == trace_name].shape[0] > 0

@retry(Exception, tries=10, delay=2, backoff=2, logger=print)
def wait_for_trace_start(trace_connection, trace_name: str) -> bool:
    """
    Polls until the trace with the specified name is detected as started.
    """
    if not trace_started(trace_connection.list_traces(), trace_name):
        raise Exception("Trace has not started yet")
    return True

def extract_session_id_from_query_begin(logs: pd.DataFrame) -> str:
    """
    Extracts the Session ID from QueryBegin events that include 'SemPy' in their Request Properties.
    """
    query_begin_events = logs[logs["Event Class"] == "QueryBegin"]
    for _, row in query_begin_events.iterrows():
        req_props = row.get("Request Properties")
        if req_props and "SemPy" in req_props:
            try:
                root = ET.fromstring(req_props)
                for child in root:
                    if child.tag.endswith("SspropInitAppName") and child.text == "SemPy":
                        return row["Session ID"]
            except Exception as e:
                print(f"Error parsing Request Properties XML: {e}")
    return None

@retry(Exception, tries=5, delay=2, backoff=2, logger=print)
def wait_for_query_end_event(trace) -> tuple[str, pd.DataFrame]:
    """
    Waits until a QueryBegin event with 'SemPy' is detected and its corresponding QueryEnd is present.
    """
    logs = trace.get_trace_logs()
    session_id = extract_session_id_from_query_begin(logs)
    if not session_id:
        raise Exception("QueryBegin event with SspropInitAppName 'SemPy' not found yet")
    query_end_events = logs[(logs["Event Class"] == "QueryEnd") & (logs["Session ID"] == session_id)]
    if query_end_events.empty:
        raise Exception("QueryEnd event for Session ID not collected yet")
    return session_id, logs

def collect_filtered_trace_logs(trace) -> pd.DataFrame:
    """
    Waits for the QueryEnd event, stops the trace, and filters logs for the relevant Session ID.
    """
    session_id, _ = wait_for_query_end_event(trace)
    logs = trace.stop()
    filtered_logs = logs[logs["Session ID"] == session_id]
    return filtered_logs

@retry(Exception, tries=10, delay=2, backoff=2, logger=print)
def check_model_online(model: dict) -> None:
    """
    Checks if the given model is online by executing a simple DAX query.
    """
    dax_query_eval_1(model)

def dax_query_eval_1(model: dict) -> None:
    """
    Executes a simple DAX query ("EVALUATE {1}") to verify connectivity and responsiveness.
    """
    try:
        fabric.evaluate_dax(model["name"], "EVALUATE {1}", workspace=model["workspace_name"])
    except Exception as e:
        raise Exception("Failed to query model") from e

def wait_for_model_to_come_online(model: dict) -> None:
    """
    Waits until the model is online by calling check_model_online.
    """
    try:
        check_model_online(model)
    except Exception as e:
        raise Exception("❌ Model failed to come online") from e

@retry(exceptions=(Exception,), tries=5, delay=5, backoff=2)
def clear_vertipaq_cache(model: dict) -> None:
    """
    Clears the VertiPaq cache via labs.clear_cache and verifies with a simple DAX query.
    """
    wait_for_model_to_come_online(model)
    try:
        labs.clear_cache(model["name"], workspace=model["workspace_name"])
        dax_query_eval_1(model)
    except Exception as e:
        print("🔄 Clearing VertiPaq cache failed; retrying...")
        fabric.refresh_tom_cache(model["workspace_name"])
        raise e
    time.sleep(5)

def run_query_with_trace(model: dict, dax_expression: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes a single DAX query under warm cache and captures:
      - query_result_df: the DataFrame result of the query
      - diagnostics_df: the storage‐engine trace logs (QueryEnd event and related)
    """

    # Clear existing traces
    fabric.create_trace_connection(model["name"], model["workspace_name"]).drop_traces()

    # Start a new trace
    trace_name = f"Trace {uuid4()}"
    with fabric.create_trace_connection(model["name"], model["workspace_name"]) as trace_conn:
        with trace_conn.create_trace(event_schema, trace_name) as trace:
            trace.start()
            wait_for_trace_start(trace_conn, trace_name)

            try:
                # <-- attempt to run the DAX; if it fails, we go to except
                query_result_df = fabric.evaluate_dax(
                    model["name"], dax_expression, workspace=model["workspace_name"]
                )
            except Exception as e:
                # If evaluation fails, still grab whatever diagnostics exist, then re‐raise.
                print(f"❌ Query execution failed: {e}")

                # collect diagnostics up to the failure point
                diagnostics_df = collect_filtered_trace_logs(trace)
                
                # Re‐raise so that orchestrate_optimization can catch it:
                raise RuntimeError(f"DAX execution error: {e}") from e

            # If we got here, the query succeeded. Now collect diagnostics.
            diagnostics_df = collect_filtered_trace_logs(trace)

    return query_result_df, diagnostics_df


# In[7]:


@log_function_calls
def load_user_resources(folder_path: str) -> dict[str, str]:
    """
    Read all files under lakehouse_mount_path + folder_path via notebookutils.
    • For PDFs: read the full file and extract plain text with PyMuPDF (fitz).
      If that fails or the PDF has no “real” text layer, return a placeholder.
    • For text files: read up to 100 MB via notebookutils.fs.head, then decode.
    Returns { filename: text_content }.
    """
    resources: dict[str, str] = {}
    try:
        mount_root = notebookutils.fs.getMountPath(lakehouse_mount_path)
        local_folder = f"{mount_root}/{folder_path}"
        files = notebookutils.fs.ls(f"file:{local_folder}")
        max_bytes = 100 * 1024 * 1024

        for f in files:
            path = f.path               # e.g. "file:/dbfs/mnt/.../folder/your.pdf"
            name = path.split("/")[-1]  # just the filename

            if name.lower().endswith(".pdf"):
                # ── PDF: read all bytes, then run through fitz to extract text
                try:
                    with notebookutils.fs.open(path, "rb") as pdf_stream:
                        pdf_bytes = pdf_stream.read()

                    # Open the PDF from raw bytes:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    all_text_pages = []
                    for page in doc:
                        text = page.get_text("text") or ""
                        all_text_pages.append(text)
                    doc.close()

                    text = "\n\n".join(all_text_pages).strip()
                    if not text:
                        # If no selectable text was found, fallback to placeholder
                        text = f"<no_text_layer_in_{name}>"

                except Exception as pdf_err:
                    text = f"<error_extracting_text_from_{name}: {pdf_err}>"

            else:
                # ── Non-PDF: read up to max_bytes, then decode
                raw_data = notebookutils.fs.head(path, max_bytes)
                if isinstance(raw_data, (bytes, bytearray)):
                    try:
                        text = raw_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_data.decode("latin-1", errors="ignore")
                else:
                    text = raw_data

            resources[name] = text

        print(f"✅ Loaded {len(resources)} resources from '{local_folder}'")
    except Exception as e:
        print(f"⚠️ Error loading resources: {e}")
    return resources


# In[8]:


@log_function_calls
def define_query_measures(query: str, model: dict) -> str:
    """
    Ensure every measure referenced in the query is declared via DEFINE MEASURE,
    including any nested measures in those definitions. Any existing DEFINE block
    (VARs, comments, etc.) is left untouched. Newly added measures appear immediately
    under the DEFINE keyword.
    """

    def normalize(name: str) -> str:
        # Strip out all non-alphanumeric characters and lowercase
        return re.sub(r"[^0-9A-Za-z]", "", name).lower()

    upper_q = query.upper()
    define_start = upper_q.find("DEFINE")
    eval_start = upper_q.find("EVALUATE")

    if 0 <= define_start < eval_start:
        # There is an existing DEFINE … EVALUATE block
        define_block = query[define_start:eval_start]
        main_query = query[eval_start:]
        has_define = True
    else:
        # No existing DEFINE; we will build one
        define_block = ""
        main_query = query
        has_define = False

    # ——————————————————————————————————————————————————————
    # 1. Find all already‐defined measures (inside the DEFINE block)
    #    We capture the raw measure‐name as it appears inside the brackets.
    def_pattern = re.compile(
        r"MEASURE\s+(?:'[^']+'|\w+)\s*\[\s*([^\]]+)\]", re.IGNORECASE
    )
    raw_existing_defs = set(def_pattern.findall(define_block))
    normalized_existing = {normalize(m) for m in raw_existing_defs}

    # 2. Build a catalog of all measures: MeasureName → (TableName, MeasureExpression)
    measures_df = fabric.list_measures(
        model["name"], workspace=model["workspace_name"]
    )
    measures_df.columns = measures_df.columns.str.replace(" ", "", regex=True)
    measures_info = {
        row["MeasureName"]: (row["TableName"], row["MeasureExpression"])
        for _, row in measures_df.iterrows()
    }

    # Create a lookup: normalized_name → actual_measure_name
    measure_lookup = {
        normalize(mname): mname for mname in measures_info.keys()
    }

    # 3. Find every token that looks like [Something] in define_block + main_query
    bracket_pattern = re.compile(r"\[([^\]]+)\]")
    def extract_bracket_tokens(text: str) -> set[str]:
        return set(bracket_pattern.findall(text))

    all_bracket_tokens = extract_bracket_tokens((define_block or "") + main_query)

    # 4. BFS over tokens to collect all missing measures (and their nested children)
    to_define: list[tuple[str, str, str]] = []
    seen = set(normalized_existing)
    queue = deque()

    # Enqueue any bracket‐token that is not already in existing defs and exists in catalog
    for tok in all_bracket_tokens:
        norm_tok = normalize(tok)
        if norm_tok not in seen and norm_tok in measure_lookup:
            queue.append(norm_tok)

    while queue:
        norm_name = queue.popleft()
        if norm_name in seen:
            continue
        # Identify actual measure name from the lookup
        actual_name = measure_lookup[norm_name]
        seen.add(norm_name)

        table_name, expr = measures_info[actual_name]
        to_define.append((actual_name, table_name, expr))

        # Enqueue any child measure references inside this expression
        for child_tok in extract_bracket_tokens(expr):
            child_norm = normalize(child_tok)
            if child_norm not in seen and child_norm in measure_lookup:
                queue.append(child_norm)

    # If there are no new measures to define, simply return the original query as‐is
    if not to_define and has_define:
        # Ensure the existing DEFINE block ends with exactly one newline
        trimmed = define_block.rstrip("\n")
        define_block_fixed = trimmed + "\n"
        return define_block_fixed + main_query
    elif not to_define and not has_define:
        return query  # no DEFINE needed

    # ——————————————————————————————————————————————————————
    # 5. Build the new DEFINE block (either inserting into existing or creating from scratch)

    # Prepare the lines for each measure we need to add.
    # We will place them immediately under the "DEFINE" line, each prefixed by a tab.
    new_measure_lines = []
    for actual_name, table_name, expr in to_define:
        line = f"\tMEASURE '{table_name}'[{actual_name}] = {expr}"
        new_measure_lines.append(line)

    if has_define:
        # Split the existing DEFINE block into lines, preserving everything
        define_lines = define_block.splitlines()
        # The first non‐empty line should be "DEFINE" (possibly with spaces/tabs)
        # Find its index, but in typical queries it's the very first line of define_block.
        first_idx = 0
        while first_idx < len(define_lines) and define_lines[first_idx].strip() == "":
            first_idx += 1

        # Now insert new measures right after the "DEFINE" line
        # Everything else (VARs, existing MEASUREs, comments) stays below.
        rebuilt = []
        rebuilt.extend(define_lines[: first_idx + 1])  # up through the "DEFINE" line
        rebuilt.extend(new_measure_lines)              # new MEASURE lines
        rebuilt.extend(define_lines[first_idx + 1 :])  # the original content below

        # Ensure a single trailing newline
        new_define_block = "\n".join(rebuilt).rstrip("\n") + "\n"

    else:
        # No existing DEFINE → create one from scratch
        # We emit:
        #   DEFINE
        #       MEASURE 'Table'[Name] = ...
        #       MEASURE 'Table2'[Name2] = ...
        define_header = "DEFINE"
        new_define_block = define_header + "\n" + "\n".join(new_measure_lines) + "\n"

    # 6. Return the newly constructed DEFINE + the original main query
    return new_define_block + main_query


# In[9]:


@log_function_calls
def extract_semantic_metadata(model: dict, dax_query: str) -> dict[str, pd.DataFrame]:
    """
    Retrieves lists of tables, columns, and relationships—but also any tables
    that filter or are filtered by those tables (recursively). Internally, this
    calls labs.get_dax_query_dependencies on `dax_query` to get a DataFrame of
    { Table Name, Column Name }, then expands that list to include any tables
    that filter (or are filtered by) the initial set via the relationships graph.
    Returns a dict with:
      - 'tables'        : pd.DataFrame (filtered to only the columns useful for an LLM)
      - 'columns'       : pd.DataFrame (filtered to only the columns useful for an LLM)
      - 'relationships' : pd.DataFrame (filtered to only the columns useful for an LLM)
    """
    metadata: dict[str, pd.DataFrame] = {}
    model_name = model["name"]
    workspace = model["workspace_name"]

    try:
        # 1) Ask SemPy which tables/columns this query actually touches
        deps_df = labs.get_dax_query_dependencies(
            dax_string=dax_query,
            dataset=model_name,
            workspace=workspace,
            show_vertipaq_stats=False
        )
        tables_used = set(deps_df["Table Name"].unique().tolist())

        # 2) Pull the full metadata from Fabric / SemPy
        full_tables = fabric.list_tables(model_name, workspace=workspace)
        full_columns = fabric.list_columns(model_name, workspace=workspace, extended=True)
        full_rels = fabric.list_relationships(model_name, workspace=workspace)

        # 3) Expand tables_used to include any tables that filter (or are filtered by)
        #    the current set, based on Cross Filtering Behavior. Repeat until no change.
        expanded_tables = set(tables_used)
        changed = True
        while changed:
            changed = False
            for _, row in full_rels.iterrows():
                from_table = row["From Table"]
                to_table = row["To Table"]
                behavior = row["Cross Filtering Behavior"]

                # OneDirection: To → From (To filters From)
                if behavior == "OneDirection":
                    if from_table in expanded_tables and to_table not in expanded_tables:
                        expanded_tables.add(to_table)
                        changed = True

                # BothDirections: From ↔ To (mutual filtering)
                elif behavior == "BothDirections":
                    if from_table in expanded_tables and to_table not in expanded_tables:
                        expanded_tables.add(to_table)
                        changed = True
                    elif to_table in expanded_tables and from_table not in expanded_tables:
                        expanded_tables.add(from_table)
                        changed = True

                # (Ignore other behaviors for filtering propagation)

        # 4) Filter tables, columns, and relationships by expanded_tables
        tables_df = full_tables[full_tables["Name"].isin(expanded_tables)].copy()
        columns_df = full_columns[full_columns["Table Name"].isin(expanded_tables)].copy()
        rels_df = full_rels[
            (full_rels["From Table"].isin(expanded_tables)) &
            (full_rels["To Table"].isin(expanded_tables))
        ].copy()

        # 5) Keep only the columns that are most relevant for an LLM to optimize DAX

        # • For tables: keep at least the table name and, if available, cardinality-related info
        desired_table_cols = [
            "Name",            # table name
            "Hidden",        # whether the table is hidden in the model
            "Description"   # number of partitions (if present)
        ]
        table_cols_to_keep = [c for c in desired_table_cols if c in tables_df.columns]
        tables_df = tables_df[table_cols_to_keep]

        # • For columns: keep table name, column name, data type, and (if available) cardinality or distribution stats
        desired_column_cols = [
            "Table Name",      # to know which table it belongs to
            "Column Name",
            "Description"     # column identifier
            "Data Type",       # data type (e.g., Text, Integer)
            # "Column Cardinality"      # nullability (if present)
        ]
        column_cols_to_keep = [c for c in desired_column_cols if c in columns_df.columns]
        columns_df = columns_df[column_cols_to_keep]

        # • For relationships: keep from/to table, from/to column, the cardinality/multiplicity,
        #   and cross‐filtering behavior
        desired_rel_cols = [
            "From Table",
            "From Column",
            "To Table",
            "To Column",
            "Multiplicity",               # e.g., "1:*", "1:1"
            "Cross Filtering Behavior",   # OneDirection / BothDirections
            "Active"                      # whether the relationship is active
        ]
        rel_cols_to_keep = [c for c in desired_rel_cols if c in rels_df.columns]
        rels_df = rels_df[rel_cols_to_keep]

        metadata["tables"] = tables_df
        metadata["columns"] = columns_df
        metadata["relationships"] = rels_df

        print(
            f"✅ Retrieved semantic metadata (filtered): "
            f"tables({len(tables_df)} cols → {len(table_cols_to_keep)} kept), "
            f"columns({len(columns_df)} cols → {len(column_cols_to_keep)} kept), "
            f"relationships({len(rels_df)} cols → {len(rel_cols_to_keep)} kept)"
        )

    except Exception as e:
        print(f"❌ Failed to extract filtered semantic metadata: {e}")
        # On failure, return empty DataFrames
        metadata = {
            "tables": pd.DataFrame(),
            "columns": pd.DataFrame(),
            "relationships": pd.DataFrame()
        }

    return metadata


# In[10]:


@log_function_calls
def summarize_server_timings(server_timings: pd.DataFrame, top_n_text: int = 3, snippet_len: int = 500) -> str:
    """
    Given the raw trace‐log DataFrame returned by run_query_with_trace, produce a concise
    summary that includes:
      • Aggregate storage‐engine CPU time and total storage‐engine duration (summing
        everything except QueryBegin and QueryEnd).
      • Breakdown of how many times each Event Class appears (except QueryBegin).
      • Average and maximum storage‐engine durations.
      • How many storage‐engine calls exceed the 90th‐percentile duration.
      • Detailed information about callbacks (rows where Text Data contains 'CallbackDataID'),
        including a count of each unique callback expression and one full example snippet.
      • A list of the top_n_text longest‐running storage‐engine calls, showing Duration, Event Class,
        and a more extensive Text Data snippet (up to snippet_len characters).
      • A list of the top_n_text most‐frequent “normalized” Text Data patterns (with counts
        and one representative raw snippet of up to snippet_len characters).
      • An estimated “formula‐engine” time (FE_time) = QueryEnd.Duration – sum(storage durations),
        if exactly one QueryEnd row is present.
      • A short “Next steps” hint list (purely actionable), enhanced with best practices from
        the Optimizing DAX book.

    Parameters:
      server_timings (pd.DataFrame): DataFrame returned by run_query_with_trace, containing
                                     at least these columns (if present):
                                      - 'Event Class' (e.g. "QueryBegin", "VertiPaqSEQueryEnd", etc.)
                                      - 'Duration'           (ms)
                                      - 'Cpu Time' or 'CpuTime' (ms)
                                      - 'Text Data'
                                      - 'StartTime', 'EndTime' (if you want FE‐time)
      top_n_text (int):         How many of the slowest & most‐frequent Text Data entries to show
                               (default = 3).
      snippet_len (int):        Maximum number of characters to include when showing Text Data
                               snippets (default = 500).

    Returns:
      str: A multi‐line summary suitable for feeding into another LLM or presenting directly.
    """

    import re

    # --- Helper: Normalize a TextData string by stripping out dates, quoted strings, and numbers ---
    def normalize_textdata(raw: str) -> str:
        if not isinstance(raw, str) or raw.strip() == "":
            return "<empty>"
        text = raw
        # 1) Replace all occurrences of DATE(  ####,  ##,  ## ) → “DATE”
        text = re.sub(r"DATE\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", "DATE", text, flags=re.IGNORECASE)
        # 2) Replace quoted literals '…' → '' (two single‐quotes)
        text = re.sub(r"'[^']*'", "''", text)
        # 3) Replace any standalone number → “0”
        text = re.sub(r"\b\d+\b", "0", text)
        # 4) Collapse multiple whitespace/newlines into single spaces, lowercase
        text = " ".join(text.split()).lower().strip()
        return text

    # 0) Verify required columns
    if "Event Class" not in server_timings.columns:
        return "⚠️ Missing required column 'Event Class'; cannot summarize."
    if "Duration" not in server_timings.columns or "Text Data" not in server_timings.columns:
        return "⚠️ Required columns ('Duration' or 'Text Data') are missing—cannot summarize."

    # 1) Split out QueryEnd rows and storage‐engine rows (exclude QueryBegin entirely)
    is_query_begin = server_timings["Event Class"] == "QueryBegin"
    query_end_df = server_timings[server_timings["Event Class"] == "QueryEnd"]
    se_df = server_timings[~is_query_begin].copy()          # includes QueryEnd + storage calls
    se_df = se_df[se_df["Event Class"] != "QueryEnd"]        # now only storage calls

    # 2) Compute estimated FE time if exactly one QueryEnd exists
    fe_time_note = "Estimated FE time: n/a"
    if len(query_end_df) == 1:
        overall_duration = float(query_end_df.iloc[0]["Duration"])
        total_storage_duration = float(se_df["Duration"].sum())
        fe_time_ms = overall_duration - total_storage_duration
        fe_time_note = f"Estimated FE time: ~{int(fe_time_ms):,} ms"
    else:
        fe_time_note = "Estimated FE time: n/a (QueryEnd row not exactly one)"

    # 3) Storage‐engine aggregates
    total_se_duration = se_df["Duration"].sum()
    # CPU time might be in “Cpu Time” or “CpuTime”
    if "Cpu Time" in se_df.columns:
        total_se_cpu = int(se_df["Cpu Time"].sum())
        cpu_str = f"Total SE CPU time: {total_se_cpu:,} ms"
    elif "CpuTime" in se_df.columns:
        total_se_cpu = int(se_df["CpuTime"].sum())
        cpu_str = f"Total SE CPU time: {total_se_cpu:,} ms"
    else:
        cpu_str = "Total SE CPU time: n/a"
    duration_str = f"Total SE duration: {int(total_se_duration):,} ms"

    # 3b) Breakdown by Event Class (e.g. “VertiPaqSEQueryEnd”, “DirectQueryEnd”, etc.)
    event_counts = se_df["Event Class"].value_counts().to_dict()

    # 3c) Average + maximum durations
    avg_duration = se_df["Duration"].mean()
    max_duration = se_df["Duration"].max()

    # 3d) How many storage calls ≥ 90th percentile
    p90 = se_df["Duration"].quantile(0.90)
    num_heavy = int((se_df["Duration"] >= p90).sum())

    # 3e) Locate CallbackDataID rows and gather details
    callback_mask = se_df["Text Data"].str.contains("CallbackDataID", na=False)
    callback_df = se_df[callback_mask].copy()
    callback_count = len(callback_df)

    # Only build the “expression‐counts” table if we actually have callbacks
    if callback_count > 0:
        # 3e.i) Count how many times each exact “Text Data” appears
        vc: pd.Series = callback_df["Text Data"].value_counts()
        # Now create a DataFrame with two columns: ["CallbackExpression","Count"]
        callback_expr_counts = vc.reset_index()
        callback_expr_counts.columns = ["CallbackExpression", "Count"]
        # 3e.ii) Take the top_n_text expressions
        top_callbacks = callback_expr_counts.head(top_n_text)

        # 3e.iii) For each of those, grab one full snippet (truncated to snippet_len)
        callback_examples: dict[str, str] = {}
        for expr in top_callbacks["CallbackExpression"]:
            full_snippet = expr.strip().replace("\n", " ")
            if len(full_snippet) > snippet_len:
                callback_examples[expr] = full_snippet[:snippet_len] + "…"
            else:
                callback_examples[expr] = full_snippet
    else:
        # No callbacks—make an empty structure so we can skip printing
        top_callbacks = pd.DataFrame(columns=["CallbackExpression", "Count"])
        callback_examples = {}

    # 4) “Slowest” top_n_text calls (by Duration → show Event Class, Duration & extended Text Data)
    slowest_df = (
        se_df[["Event Class", "Duration", "Text Data"]]
        .sort_values("Duration", ascending=False)
        .head(top_n_text)
        .reset_index(drop=True)
    )

    # 5) “Most‐frequent” normalized Text Data patterns
    se_df["Normalized"] = se_df["Text Data"].apply(normalize_textdata)
    norm_counts = se_df["Normalized"].value_counts().head(top_n_text).reset_index()
    norm_counts.columns = ["Normalized", "Count"]

    # 5b) For each top normalized pattern, grab one raw snippet (longer)
    rep_snippets: dict[str, str] = {}
    for norm_val in norm_counts["Normalized"]:
        example_raw = se_df.loc[se_df["Normalized"] == norm_val, "Text Data"].iloc[0]
        cleaned = example_raw.strip().replace("\n", " ")
        if len(cleaned) > snippet_len:
            rep_snippets[norm_val] = cleaned[:snippet_len] + "…"
        else:
            rep_snippets[norm_val] = cleaned

    # 6) Assemble the summary text
    lines: list[str] = [
        "🗒️ **Server‐Timings Summary**",
        "",
        f"• {cpu_str}    {duration_str}",
        f"• Breakdown by Event Class (storage calls): {{ { {k: int(v) for k, v in event_counts.items()} } }}",
        f"• Avg storage‐engine call duration: {avg_duration:.2f} ms    Max storage‐engine call duration: {max_duration:.2f} ms",
        f"• Storage‐engine calls ≥ 90th percentile (≥ {p90:.2f} ms): {num_heavy}",
        "",
        f"• CallbackDataID hits: {callback_count}"
    ]

    # 6a) If we have callbacks, list the top expressions
    if callback_count > 0:
        lines.append("  • Top CallbackDataID expressions (Count, representative snippet):")
        for idx, row in top_callbacks.iterrows():
            expr = row["CallbackExpression"]
            cnt = int(row["Count"])
            example_snip = callback_examples.get(expr, "")
            lines.append(f"    {idx+1}. [Count={cnt}] «{example_snip}»")
        lines.append("")

    # 7) Append the slowest calls with extended snippets
    lines.append(f"• {top_n_text} Slowest storage‐engine calls (Event Class, Duration, Text Data):")
    for idx, row in slowest_df.iterrows():
        ev = row["Event Class"]
        d = row["Duration"]
        raw = row["Text Data"].strip().replace("\n", " ")
        if len(raw) > snippet_len:
            raw_trunc = raw[:snippet_len] + "…"
        else:
            raw_trunc = raw
        lines.append(f"   {idx+1}. [{ev}] [{d:.2f} ms] «{raw_trunc}»")
    lines.append("")

    # 8) Append the most‐frequent normalized patterns with extended snippets
    lines.append(f"• {top_n_text} Most‐frequent “normalized” calls (Count, representative Text Data):")
    for idx, row in norm_counts.iterrows():
        cnt = int(row["Count"])
        norm_val = row["Normalized"]
        snippet = rep_snippets[norm_val]
        lines.append(f"   {idx+1}. [Count={cnt}] «{snippet}»")
    lines.append("")
    lines.append(f"• {fe_time_note}")
    lines.append("")
    return "\n".join(lines)


# In[11]:


@log_function_calls
def optimize_query_with_fabric_ai(
    original_query: str,
    iteration: int,
    server_timings: pd.DataFrame,
    metadata: dict[str, pd.DataFrame],
    resources: dict[str, str]
) -> str:
    """
    """

    # Helper to strip markdown fences around DAX
    def _clean_dax(raw: str) -> str:
        cleaned = re.sub(r"^(?:[Dd][Aa][Xx]\s*)?\n?", "", raw).strip()
        return re.sub(r"\n?$", "", cleaned).strip()

    # Summarize history
    diag_csv = server_timings.to_csv(index=False) if not server_timings.empty else ""

    # Semantic metadata excerpts
    try:
        tbl_df = metadata.get("tables", pd.DataFrame())
        col_df = metadata.get("columns", pd.DataFrame())
        rel_df = metadata.get("relationships", pd.DataFrame())

        # Convert each DataFrame to CSV so that the prompt includes the full contents
        tbl_csv = tbl_df.to_csv(index=False)
        col_csv = col_df.to_csv(index=False)
        rel_csv = rel_df.to_csv(index=False)

        tables_full = f"Tables ({len(tbl_df)} rows):\n{tbl_csv}"
        columns_full = f"Columns ({len(col_df)} rows):\n{col_csv}"
        rels_full = f"Relationships ({len(rel_df)} rows):\n{rel_csv}"

        model_def_excerpt = f"{tables_full}\n{columns_full}\n{rels_full}"
    except Exception:
        model_def_excerpt = "<unable to fetch semantic metadata>"

    # Resources excerpt
    if resources:
        resources_full = ""
        for name, text in resources.items():
            resources_full += f"--- {name} ---\n{text.strip()}\n\n"
    else:
        resources_full = "<no resources>"

    # Additional context
    if additional_context:
        additional_context_block=f"""
            Here is some additional context that might help:
            {additional_context}
        """
    else: 
        additional_context_block = ""
        
        # Here are helpful resources to aid you in optimizing DAX queries and measures.
        # *Helpful Resources Start*
        # {resources_full}
        # *Helpful Resources End*

    # Construct coach prompt
    coach_prompt = f"""
        You are a Semantic Model and DAX expert whose job is to analyze a DAX‐query and give optimization suggestions.
        Based on the DAX query and model metadata (tables/columns/relationships) below, provide specific, actionable suggestions for what changes can be made to improve performance.
        
        Original Query:
        *Query Start*
        {original_query}
        *Query End*

        *Model Metadata Start*
        {model_def_excerpt}
        *Model Metadata End*

        {summarize_server_timings(server_timings)}

        {additional_context_block}

        Instructions:
        • Ensure that your suggestions are valid DAX sytnax patterns. For example, you can not move a measure to a CALCULATE filter arguement because it is not a table.
        • Look for information on https://www.daxpatterns.com/ and https://sqlbi.com/ to identify patterns and optimization techniques.
        • After analyzing the query, give your optimization recommendation based on your finding.
        • **Before finalizing**, make sure that your recommendation is valid.
        • Be concise and specific with your recommendation. Let the user know which part of the query they should update and what they should update it with.
        • Keep your answer targetted to the optimization recommendation.
        • Be as creative as you need to be. It is completely acceptable to recommend a totally different pattern (for instance, replacing DISTINCTCOUNT with SUMX(VALUES(...),1)), but make sure that it is semantically equivalent and it is a valid DAX expression.
        • Suggest recommendations that adjust ONLY adjust MEASURE definitions; do not recommend modifying any other part of query.
        • Assume that all of the filter variables are needed and cannot be changed or modified.
        • Asumme that we don't have the ability to edit the underlying data model. This means we should recommend changes to the MEASURES instead of telling the user to add a new column to the model.
        • Use the iteration history (including any failure summaries and error texts) to avoid recommending past mistakes. That is, if Iter 1 or Iter 2 failed with a “Table variable …” error, inspect those error messages and do not recommend the same pattern again. Instead, choose an alternate approach.
    """.strip()
    # Generate coach response via AI
    return coach_prompt


# In[16]:


defined_dax_query = define_query_measures(target_dax_query, model)
run_query_with_trace(model, defined_dax_query)
clear_vertipaq_cache(model)
baseline_df, baseline_diagnostics = run_query_with_trace(model, defined_dax_query)
metadata = extract_semantic_metadata(model, target_dax_query)
resources = load_user_resources(resource_folder_path)
tbl_df = metadata.get("tables", pd.DataFrame())
col_df = metadata.get("columns", pd.DataFrame())
rel_df = metadata.get("relationships", pd.DataFrame())


# In[22]:


display(rel_df )


# In[13]:


# # 1) Ask LLM for a new optimized query
# recommendations = optimize_query_with_fabric_ai(
#     original_query=defined_dax_query,
#     iteration=1,
#     server_timings=baseline_diagnostics,
#     metadata=metadata,
#     resources=resources
# ).strip()

# df_coach = pd.DataFrame([{"dummy": ""}])
# df_coach["recommendations_summary"] = df_coach.ai.generate_response(
#     prompt=recommendations,
#     conf=Conf(
#         temperature=ai_temperature,           
#         model_deployment_name=ai_model
#     )
# )
# recommendations_summary = str(df_coach.loc[0, "recommendations_summary"]).strip()

# print(recommendations_summary)

