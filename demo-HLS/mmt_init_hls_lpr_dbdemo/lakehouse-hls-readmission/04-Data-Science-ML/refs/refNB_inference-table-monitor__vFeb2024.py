# Databricks notebook source
# MAGIC %md # Inference Table Analysis Notebook
# MAGIC
# MAGIC #### About this notebook
# MAGIC This starter notebook is intended to be used with **Databricks Model Serving** endpoints which have the *Inference Table* feature enabled.</br>
# MAGIC This notebook has three high-level purposes:
# MAGIC 1. Process logged requests and responses by converting raw JSON payloads to Spark data types.
# MAGIC 2. Join requests with relevant tables, such as labels or business metrics.
# MAGIC 3. Run Databricks Lakehouse Monitoring on the resulting table to produce data and model quality/drift metrics.
# MAGIC
# MAGIC #### How to run the notebook
# MAGIC In order to use the notebook, you should populate the parameters in the **Parameters** section below (entered in the following cell) with the relevant information.</br>
# MAGIC For best results, run this notebook on any cluster running **Databricks Runtime 12.2LTS or higher**.
# MAGIC
# MAGIC #### Scheduling
# MAGIC Feel free to run this notebook manually to test out the parameters; when you're ready to run it in production, you can schedule it as a recurring job.</br>
# MAGIC Note that in order to keep this notebook running smoothly and efficiently, we recommend running it at least **once a week** to keep output tables fresh and up to date.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC Install the Python client and restart Python.

# COMMAND ----------

# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.6-py3-none-any.whl"

# COMMAND ----------

# This step is necessary to reset the environment with our newly installed wheel.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Parameters
# MAGIC This section contains all of the parameters needed to run this notebook successfully. Please be sure to provide the correct information and test that the notebook works end-to-end before scheduling it at a regular interval.
# MAGIC
# MAGIC **Required parameters**:
# MAGIC - `ENDPOINT_NAME`: Name of the model serving endpoint
# MAGIC - `PROBLEM_TYPE`: ML problem type, for `problem_type` parameter of `create_monitor`. One of `"classification"` or `"regression"`.
# MAGIC
# MAGIC **Monitoring parameters**:
# MAGIC - `GRANULARITIES`: List of window granularities, for `granularities` parameter of `create_monitor`.
# MAGIC - `SLICING_EXPRS`: List of expressions to slice data with, for `slicing_exprs` parameter of `create_monitor`.
# MAGIC - `CUSTOM_METRICS`: List of custom metrics to calculate during monitoring analysis, for `custom_metrics` parameter of `create_monitor`.
# MAGIC - `BASELINE_TABLE`: Name of table containing baseline data, for `baseline_table_name` parameter of `create_monitor`.
# MAGIC
# MAGIC **Table parameters**:
# MAGIC - `PREDICTION_COL`: Name of column to store predictions in the generated tables.
# MAGIC - `LABEL_COL`: Name of column to store joined labels in the generated tables.
# MAGIC - `REQUEST_FIELDS`: A list of `StructField`'s representing the schema of the endpoint's inputs, which is necessary to unpack the raw JSON strings. Must have one struct field per input column. If not provided, will attempt to infer from the endpoint's logged models' signatures.
# MAGIC - `RESPONSE_FIELD`: A `StructField` representing the schema of the endpoint's output, which is necessary to unpack the raw JSON strings. Must have exactly one struct field with name as `PREDICTION_COL`. If not provided, will attempt to infer from the endpoint's logged models' signatures.
# MAGIC - `JOIN_TABLES`: An optional specification of Unity Catalog tables to join the requests with. Each table must be provided in the form of a tuple containing: `(<table name>, <list of columns to join>, <list of columns to use for equi-join>)`.
# MAGIC - `FILTER_EXP`: Optionally filter the requests based on a SQL expression that can be used in a `WHERE` clause. Use this if you want to persist and monitor a subset of rows from the logged data.
# MAGIC - `PROCESSING_WINDOW_DAYS`: A window size that restricts the age of data processed since the last run of this notebook. Data older than this window will be ignored if it has not already been processed, including joins specified by `JOIN_TABLES`.
# MAGIC - `SPECIAL_CHAR_COMPATIBLE`: Optionally set Delta table properties in order to handle column names with special characters like spaces. Set this to False if you want to maintain the ability to read these tables on older runtimes or by external systems.

# COMMAND ----------

from pyspark.sql import functions as F, types as T


"""
Required parameters in order to run this notebook.
"""
ENDPOINT_NAME = None       # Name of the model serving endpoint
PROBLEM_TYPE = None        # ML problem type, one of "classification"/"regression"

# Validate that all required inputs have been provided
if None in [ENDPOINT_NAME, PROBLEM_TYPE]:
    raise Exception("Please fill in the required information for endpoint name and problem type.")

"""
Optional parameters to control monitoring analysis.
"""
# Monitoring configuration parameters. For help, use the command help(lm.create_monitor).
GRANULARITIES = ["1 day"]                        # Window sizes to analyze data over
SLICING_EXPRS = None                             # Expressions to slice data with
CUSTOM_METRICS = None                            # A list of custom metrics to compute
BASELINE_TABLE = None                            # Baseline table name, if any, for computing baseline drift

"""
Optional parameters to control processed tables.
"""
PREDICTION_COL = "predictions"                   # What to name the prediction column in the processed table
LABEL_COL = None                                 # Name of columns holding labels. Change this if you join your own labels via JOIN_TABLES above.
# The endpoint input schema, e.g.: [T.StructField("feature_one", T.StringType()), T.StructField("feature_two", T.IntegerType())]
# If not provided, will attempt to infer request fields from one of the logged models' signatures.
REQUEST_FIELDS = None
# The endpoint output schema, e.g.: T.StructField(PREDICTION_COL, T.IntegerType()). Field name must be PREDICTION_COL.
# If not provided, will attempt to infer response field from one of the logged models' signatures.
RESPONSE_FIELD = None
# For each table to join with, provide a tuple of: (table name, columns to join, columns to use for equi-join)
# For the equi-join key columns, this will be either "client_request_id" or a combination of features that uniquely identifies an example to be joined with.
# Tables must be registered in Unity Catalog
JOIN_TABLES = [
    # Example: ("labels_table", ["labels", "client_request_id"], ["client_request_id"])
]

# Optionally filter the request logs based on an expression that can be used in a WHERE clause.
# For example: "timestamp > '2022-10-10' AND age >= 18"
FILTER_EXP = None

# If the request/response schema have special characters (such as spaces) in the column names,
# this flag upgrades the minimum Delta reader/writer version and sets the column mapping mode
# to use names. This in turn reduces compatibility with reading this table on older runtimes
# or by external systems. If your column names have no special characters, set this to False.
SPECIAL_CHAR_COMPATIBLE = True

# In order to bound performance of each run, we limit the window of data to be processed from the raw request tables.
# This means that data older than PROCESSING_WINDOW_DAYS will be ignored *if it has not already been processed*.
# If join data (specified by JOIN_TABLES) arrives later than PROCESSING_WINDOW_DAYS, it will be ignored.
# Increase this parameter to ensure you process more data; decrease it to improve performance of this notebook.
PROCESSING_WINDOW_DAYS = 365

# COMMAND ----------

# MAGIC %md ### Imports

# COMMAND ----------

import dataclasses
import json
import requests
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.sql.utils import AnalysisException

import databricks.lakehouse_monitoring as lm

# COMMAND ----------

# MAGIC %md ### Helper functions
# MAGIC
# MAGIC Helper functions to read, process, and write the data requests.

# COMMAND ----------

"""
Conversion helper functions.
"""
def convert_to_record_json(json_str: str) -> str:
    """
    Converts records from the four accepted JSON formats for Databricks
    Model Serving endpoints into a common, record-oriented
    DataFrame format which can be parsed by the PySpark function from_json.
    
    :param json_str: The JSON string containing the request or response payload.
    :return: A JSON string containing the converted payload in record-oriented format.
    """
    try:
        request = json.loads(json_str)
    except json.JSONDecodeError:
        return json_str
    output = []
    if isinstance(request, dict):
        obj_keys = set(request.keys())
        if "dataframe_records" in obj_keys:
            # Record-oriented DataFrame
            output.extend(request["dataframe_records"])
        elif "dataframe_split" in obj_keys:
            # Split-oriented DataFrame
            dataframe_split = request["dataframe_split"]
            output.extend([dict(zip(dataframe_split["columns"], values)) for values in dataframe_split["data"]])
        elif "instances" in obj_keys:
            # TF serving instances
            output.extend(request["instances"])
        elif "inputs" in obj_keys:
            # TF serving inputs
            output.extend([dict(zip(request["inputs"], values)) for values in zip(*request["inputs"].values())])
        elif "predictions" in obj_keys:
            # Predictions
            output.extend([{PREDICTION_COL: prediction} for prediction in request["predictions"]])
        return json.dumps(output)
    else:
        # Unsupported format, pass through
        return json_str


@F.pandas_udf(T.StringType())
def json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
    """A UDF to apply the JSON conversion function to every request/response."""
    return json_strs.apply(convert_to_record_json)


"""
Request helper functions.
"""
def get_endpoint_status(endpoint_name: str) -> Dict:
    """
    Fetches the status and config of and endpoint using the `serving-endpoints` REST endpoint.
    
    :param endpoint_name: Name of the serving endpoint
    :return: Dict containing JSON status response
    
    """
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
    headers = {"Authorization": f"Bearer {token}"}
    request = dict(name=endpoint_name)
    response = requests.get(url, json=request, headers=headers)
    
    # Check for unauthorization errors due to PAT token
    if "unauthorized" in response.text.lower():
        raise Exception(
          f"Unable to retrieve status for endpoint '{endpoint_name}'. "
          "If you are an admin, please try using a cluster in Single User security mode."
        )
        
    response_json = response.json()

    # Verify that Model Serving is enabled.
    if "state" not in response_json:
        raise Exception(f"Model Serving is not enabled for endpoint {endpoint_name}. "
                        "Please validate the status of this endpoint before running this notebook.")

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response_json["config"] or not response_json["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. "
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response_json


def delimit_identifier(identifier: str) -> str:
    """
    Delimits an identifier name using backticks to handle special characters.
    For example, "endpoint-inference-table" becomes `endpoint-inference-table`.

    :param identifier: Name of the identifier to delimit
    :return: Delimited identifier name
    """
    return f"`{identifier.replace('`', '``')}`"


@dataclasses.dataclass
class InferenceTableConfig:
    """
    Data class to store configuration info for Inference Tables.
    """
    # Delimited name of the catalog.
    catalog: str
    # Delimited name of the schema.
    schema: str
    # Delimited and fully qualified name of the payload table.
    payload_table_name: str
    # Delimited and fully qualified name of the unpacked requests table.
    unpacked_requests_table_name: str
    # Delimited and fully qualified name of the processed requests table.
    processed_requests_table_name: str


def collect_inference_table_config(endpoint_name: str) -> InferenceTableConfig:
    """
    Collects the Inference Table configuration necessary to reference all tables used by this notebook:
        - The payload table configured by the endpoint
        - The unpacked requests table produced by this notebook
        - The processed requests table produced by this notebook
    
    Note that this relies on fetching a Personal Access Token (PAT) from the
    runtime context, which can fail in certain scenarios if run by an admin on a shared cluster.
    If you are experiencing issues, you can try switch to a Single User cluster.
        
    :param endpoint_name: Name of the serving endpoint
    :return: InferenceTableConfig containing Unity Catalog identifiers
    """
    response_json = get_endpoint_status(endpoint_name=endpoint_name)

    auto_capture_config = response_json["config"]["auto_capture_config"]
    catalog = auto_capture_config["catalog_name"]
    schema = auto_capture_config["schema_name"]
    table_name_prefix = auto_capture_config["table_name_prefix"]

    # These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
    payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
    unpacked_table_name = f"{table_name_prefix}_unpacked"
    processed_table_name = f"{table_name_prefix}_processed"

    # Escape identifiers with backticks before returning to handle special characters
    delimited_catalog = delimit_identifier(catalog)
    delimited_schema = delimit_identifier(schema)
    delimited_qualified_schema = f"{delimited_catalog}.{delimited_schema}"
    return InferenceTableConfig(
        catalog=delimited_catalog,
        schema=delimited_schema,
        payload_table_name=f"{delimited_qualified_schema}.{delimit_identifier(payload_table_name)}",
        unpacked_requests_table_name=f"{delimited_qualified_schema}.{delimit_identifier(unpacked_table_name)}",
        processed_requests_table_name=f"{delimited_qualified_schema}.{delimit_identifier(processed_table_name)}",
    )


def get_served_models(endpoint_name: str) -> List[Dict]:
    """
    Fetches the list of models being served by an endpoint.
    
    :param endpoint_name: Name of the serving endpoint
    :return: List of Dicts for each served model
    """
    response_json = get_endpoint_status(endpoint_name=endpoint_name)
    if "config" not in response_json and "pending_config" not in response_json:
        raise Exception(f"Unable to find any config for endpoint '{endpoint_name}'.")
    config = response_json["config"] if "config" in response_json else response_json["pending_config"]
    if "served_models" not in config or len(config["served_models"]) == 0:
        raise Exception(f"Unable to find any served models for endpoint '{endpoint_name}'.")
    
    served_models = config["served_models"]
    return served_models


def convert_numpy_dtype_to_spark(dtype: np.dtype) -> T.DataType:
    """
    Converts the input numpy type to a Spark type.
    
    :param dtype: The numpy type
    :return: The Spark data type
    """
    NUMPY_SPARK_DATATYPE_MAPPING = {
        np.byte: T.LongType(),
        np.short: T.LongType(),
        np.intc: T.LongType(),
        np.int_: T.LongType(),
        np.longlong: T.LongType(),
        np.ubyte: T.LongType(),
        np.ushort: T.LongType(),
        np.uintc: T.LongType(),
        np.half: T.DoubleType(),
        np.single: T.DoubleType(),
        np.float_: T.DoubleType(),
        np.bool_: T.BooleanType(),
        np.object_: T.StringType(),
        np.str_: T.StringType(),
        np.unicode_: T.StringType(),
        np.bytes_: T.BinaryType(),
        np.timedelta64: T.LongType(),
        np.datetime64: T.TimestampType(),
    }
    for source, target in NUMPY_SPARK_DATATYPE_MAPPING.items():
        if np.issubdtype(dtype, source):
            return target
    raise ValueError(f"Unsupported numpy dtype: {dtype}")


def infer_request_response_fields(endpoint_name: str) -> Tuple[Optional[List[T.StructField]], Optional[T.StructField]]:
    """
    Infers the request and response schema of the endpoint by loading the signature
    of the first available served model with a logged signature
    and extracting the Spark struct fields for each respective schema.
    
    Note that if the signature varies across served models within the endpoint, this will only
    use the first available one; if you need to handle multiple signatures, please use the
    REQUEST_FIELDS and RESPONSE_FIELD parameters at the top of the notebook.
    
    Raises an error if:
    - The endpoint doesn't exist
    - The endpoint doesn't have a served model with a logged signature
    
    :param endpoint_name: Name of the serving endpoint to infer schemas for
    :return: A tuple containing a list of struct fields for the request schema, and a
             single struct field for the response. Either element may be None if this
             endpoint's models' signatures did not contain an input or output signature, respectively.
    """
    # Load the first model (with a logged signature) being served by this endpoint
    served_models = get_served_models(endpoint_name=endpoint_name)
    signature = None
    for served_model in served_models:
        model_name = served_model["model_name"]
        model_version = served_model["model_version"]
        loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        if loaded_model.metadata.signature is not None:
            signature = loaded_model.metadata.signature
            break

    if signature is None:
        raise Exception("One of REQUEST_FIELDS or RESPONSE_FIELD was not specified, "
                        "but endpoint has no served models with a logged signature. Please define the schemas "
                        "in the Parameters section of this notebook.")
    
    # Infer the request schema from the model signature
    request_fields = None if signature.inputs is None else signature.inputs.as_spark_schema().fields
    if signature.outputs is None:
        response_field = None
    else:
        # Get the Spark datatype for the model output
        model_output_schema = signature.outputs
        if model_output_schema.is_tensor_spec():
            if len(model_output_schema.input_types()) > 1:
                raise ValueError("Models with multiple outputs are not supported for monitoring")
            output_type = convert_numpy_dtype_to_spark(model_output_schema.numpy_types()[0])
        else:
            output_type = model_output_schema.as_spark_schema()
            if isinstance(output_type, T.StructType):
                if len(output_type.fields) > 1:
                    raise ValueError(
                        "Models with multiple outputs are not supported for monitoring")
                else:
                    output_type = output_type[0].dataType
        response_field = T.StructField(PREDICTION_COL, output_type)
        
    return request_fields, response_field
    

def process_requests(requests_raw: DataFrame, request_fields: List[T.StructField], response_field: T.StructField) -> DataFrame:
    """
    Takes a stream of raw requests and processes them by:
        - Unpacking JSON payloads for requests and responses
        - Exploding batched requests into individual rows
        - Converting Unix epoch millisecond timestamps to be Spark TimestampType
        
    :param requests_raw: DataFrame containing raw requests. Assumed to contain the following columns:
                            - `request`
                            - `response`
                            - `timestamp_ms`
    :param request_fields: List of StructFields representing the request schema
    :param response_field: A StructField representing the response schema
    :return: A DataFrame containing processed requests
    """
    # Convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = requests_raw \
        .withColumn(TIMESTAMP_COL, (F.col("timestamp_ms") / 1000).cast(T.TimestampType())) \
        .drop("timestamp_ms")

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped \
        .withColumn(MODEL_ID_COL, F.concat(F.col("request_metadata").getItem("model_name"), F.lit("_"), F.col("request_metadata").getItem("model_version"))) \
        .drop("request_metadata")

    # Rename the date column to avoid collisions with features.
    requests_dated = requests_identified.withColumnRenamed("date", DATE_COL)

    # Consolidate and unpack JSON.
    request_schema = T.ArrayType(T.StructType(request_fields))
    response_schema = T.ArrayType(T.StructType([response_field]))
    requests_unpacked = requests_dated \
        .withColumn("request", json_consolidation_udf(F.col("request"))) \
        .withColumn("response", json_consolidation_udf(F.col("response"))) \
        .withColumn("request", F.from_json(F.col("request"), request_schema)) \
        .withColumn("response", F.from_json(F.col("response"), response_schema))

    # Explode batched requests into individual rows.
    DB_PREFIX = "__db"
    requests_exploded = requests_unpacked \
        .withColumn(f"{DB_PREFIX}_request_response", F.arrays_zip(F.col("request"), F.col("response"))) \
        .withColumn(f"{DB_PREFIX}_request_response", F.explode(F.col(f"{DB_PREFIX}_request_response"))) \
        .select(F.col("*"), F.col(f"{DB_PREFIX}_request_response.request.*"), F.col(f"{DB_PREFIX}_request_response.response.*")) \
        .drop(f"{DB_PREFIX}_request_response", "request", "response")

    # Generate an example ID so we can de-dup each row later when upserting results.
    requests_processed = requests_exploded \
        .withColumn(EXAMPLE_ID_COL, F.expr("uuid()"))
    
    return requests_processed


"""
Table helper functions.
"""
def initialize_table(fully_qualified_table_name: str, schema: T.StructType, special_char_compatible: bool = False) -> None:
    """
    Initializes an output table with the Delta Change Data Feed.
    All tables are partitioned by the date column for optimal performance.
    
    :param fully_qualified_table_name: Fully qualified name of the table to initialize, like "{catalog}.{schema}.{table}"
    :param schema: Spark schema of the table to initialize
    :param special_char_compatible: Boolean to determine whether to upgrade the min reader/writer
                                    version of Delta and use column name mapping mode. If True,
                                    this allows for column names with spaces and special characters, but
                                    it also prevents these tables from being read by external systems
                                    outside of Databricks. If the latter is a requirement, and the model
                                    requests contain feature names containing only alphanumeric or underscore
                                    characters, set this flag to False.
    :return: None
    """
    dt_builder = DeltaTable.createIfNotExists(spark) \
        .tableName(fully_qualified_table_name) \
        .addColumns(schema) \
        .partitionedBy(DATE_COL)
    
    if special_char_compatible:
        dt_builder = dt_builder \
            .property("delta.enableChangeDataFeed", "true") \
            .property("delta.columnMapping.mode", "name") \
            .property("delta.minReaderVersion", "2") \
            .property("delta.minWriterVersion", "5")
    
    dt_builder.execute()
    
    
def read_requests_as_stream(fully_qualified_table_name: str) -> DataFrame:
    """
    Reads the endpoint's Inference Table to return a single streaming DataFrame
    for downstream processing that contains all request logs.

    If the endpoint is not logging data, will raise an Exception.
    
    :param fully_qualified_table_name: Fully qualified name of the payload table
    :return: A Streaming DataFrame containing request logs
    """
    try:
        return spark.readStream.format("delta").table(fully_qualified_table_name)
    except AnalysisException:
        raise Exception("No payloads have been logged to the provided Inference Table location. "
                        "Please be sure Inference Tables is enabled and ready before running this notebook.")


def optimize_table_daily(delta_table: DeltaTable, vacuum: bool = False) -> None:
    """
    Runs OPTIMIZE on the provided table if it has not been run already today.
    
    :param delta_table: DeltaTable object representing the table to optimize.
    :param vacuum: If true, will VACUUM commit history older than default retention period.
    :return: None
    """
    # Get the full history of the table.
    history_df = delta_table.history()
    
    # Filter for OPTIMIZE operations that have happened today.
    optimize_today = history_df.filter(F.to_date(F.col("timestamp")) == F.current_date())
    if optimize_today.count() == 0:
        # No OPTIMIZE has been run today, so initiate it.
        delta_table.optimize().executeCompaction()
        
        # Run VACUUM if specified.
        if vacuum:
            delta_table.vacuum()

# COMMAND ----------

# MAGIC %md ### Initializations
# MAGIC
# MAGIC Initialize the Spark configurations and variables used by this notebook.

# COMMAND ----------

# Enable automatic schema evolution if we decide to add columns later.
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", True)

# Set column names used in the tables produced by the notebook.
TIMESTAMP_COL = "__db_timestamp"
DATE_COL = "__db_date"
MODEL_ID_COL = "__db_model_id"
EXAMPLE_ID_COL = "__db_example_id"

# If request/response schema is None, try to infer them from the model signature.
if REQUEST_FIELDS is None or RESPONSE_FIELD is None:
    inferred_request_fields, inferred_response_field = infer_request_response_fields(endpoint_name=ENDPOINT_NAME)
    REQUEST_FIELDS = REQUEST_FIELDS or inferred_request_fields
    RESPONSE_FIELD = RESPONSE_FIELD or inferred_response_field
    if REQUEST_FIELDS is None:
        raise Exception("No REQUEST_FIELDS was provided, and no input signature was logged for any model served by the endpoint. "
                        "Please explicitly define REQUEST_FIELDS in the Parameters section.")
    if RESPONSE_FIELD is None:
        raise Exception("No RESPONSE_FIELD was provided, and no output signature was logged for any model served by the endpoint. "
                        "Please explicitly define RESPONSE_FIELD in the Parameters section.")

# COMMAND ----------

# MAGIC %md ### Process request logs

# COMMAND ----------

# MAGIC %md #### Collect Inference Table configuration
# MAGIC
# MAGIC First, we fetch the configurated Unity Catalog location for the payload table produced by the endpoint.</br></br>**NOTE**: this request may fail if you are a workspace admin and are not using the **Single User** cluster security mode.</br>In such cases, please switch to a Single User cluster.

# COMMAND ----------

inference_table_config = collect_inference_table_config(endpoint_name=ENDPOINT_NAME)

payload_table_name = inference_table_config.payload_table_name
unpacked_requests_table_name = inference_table_config.unpacked_requests_table_name
processed_requests_table_name = inference_table_config.processed_requests_table_name

display(pd.DataFrame({
  "Payload table name": payload_table_name,
  "Unpacked requests table name": unpacked_requests_table_name,
  "Processed requests table name": processed_requests_table_name,
}, index=[0]))

# COMMAND ----------

# MAGIC %md #### Process and persist requests
# MAGIC
# MAGIC Next, we apply our UDF to every row and persist the structured requests to a Delta table.
# MAGIC We also do some additional processing, e.g. to convert timestamp milliseconds to `TimestampType`
# MAGIC
# MAGIC If you encounter a `StreamingQueryException` at this step, try deleting the checkpoint directory using `dbutils.fs.rm(checkpoint_path, recurse=True)` and restarting the notebook.

# COMMAND ----------

# Read the requests as a stream so we can incrementally process them.
requests_raw = read_requests_as_stream(fully_qualified_table_name=payload_table_name)

# Unpack the requests.
requests_unpacked = process_requests(
    requests_raw=requests_raw,
    request_fields=REQUEST_FIELDS,
    response_field=RESPONSE_FIELD,
)

# Filter the requests if an expression was provided.
if FILTER_EXP is not None:
    requests_unpacked = requests_unpacked.filter(FILTER_EXP)

# Initialize the processed requests table so we can enable CDF and reader/writer versions for compatibility.
initialize_table(
    fully_qualified_table_name=unpacked_requests_table_name,
    schema=requests_unpacked.schema,
    special_char_compatible=SPECIAL_CHAR_COMPATIBLE,
)

# Persist the requests stream, with a defined checkpoint path for this table.
checkpoint_path = f"dbfs:/payload-logging/{ENDPOINT_NAME}/checkpoint"
requests_stream = requests_unpacked.writeStream \
    .trigger(once=True) \
    .format("delta") \
    .partitionBy(DATE_COL) \
    .outputMode("append") \
    .option("checkpointLocation", checkpoint_path) \
    .toTable(unpacked_requests_table_name)
requests_stream.awaitTermination()

# COMMAND ----------

# MAGIC %md ### Join with labels/data
# MAGIC
# MAGIC In this section, we optionally join the requests with other tables that contain relevant data, such as inference labels or other business metrics.</br>Note that this cell should be run even if there are no tables yet to join with.
# MAGIC
# MAGIC In order to persist the join, we do an `upsert`: insert new rows if they don't yet exist, or update them if they do.</br>This allows us to attach late-arriving information to requests.

# COMMAND ----------

# Load the unpacked requests and join them with any specified tables.
# Filter requests older than PROCESSING_WINDOW_DAYS for optimal preformance.
requests_processed = spark.table(unpacked_requests_table_name) \
    .filter(f"CAST({DATE_COL} AS DATE) >= current_date() - (INTERVAL {PROCESSING_WINDOW_DAYS} DAYS)")
for table_name, preserve_cols, join_cols in JOIN_TABLES:
    join_data = spark.table(table_name)
    requests_processed = requests_processed.join(join_data.select(preserve_cols), on=join_cols, how="left")

# Drop columns that we don't need for monitoring analysis.
requests_cleaned = requests_processed.drop("status_code", "sampling_fraction", "client_request_id", "databricks_request_id")
    
# Initialize the processed requests table so we can always use Delta merge operations for joining data.
initialize_table(
    fully_qualified_table_name=processed_requests_table_name,
    schema=requests_cleaned.schema,
    special_char_compatible=SPECIAL_CHAR_COMPATIBLE,
)

# Upsert rows into the existing table, identified by their example_id.
# Use date in the merge condition in order to allow Spark to dynamically prune partitions.
merge_cols = [DATE_COL, EXAMPLE_ID_COL]
merge_condition = " AND ".join([f"existing.{col} = updated.{col}" for col in merge_cols])
processed_requests_delta_table = DeltaTable.forName(spark, processed_requests_table_name)
processed_requests_delta_table.alias("existing") \
    .merge(source=requests_cleaned.alias("updated"), condition=merge_condition) \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()

# COMMAND ----------

# MAGIC %md ### Monitor the inference table
# MAGIC
# MAGIC In this step, we create a monitor on our inference table by using the `create_monitor` API. If the monitor already exists, we pass the same parameters to `update_monitor`. In steady state, this should result in no change to the monitor.
# MAGIC
# MAGIC Afterwards, we queue a metric refresh so that the monitor analyzes the latest processed requests.
# MAGIC
# MAGIC See the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/index.html)|[Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/index)) for more details on the parameters and the expected usage.

# COMMAND ----------

# Use the catalog/schema of the payload table as the output schema.
output_schema_name = f"{inference_table_config.catalog}.{inference_table_config.schema}"

try:
    info = lm.create_monitor(
        table_name=processed_requests_table_name,
        profile_type=lm.InferenceLog(
            timestamp_col=TIMESTAMP_COL,
            granularities=GRANULARITIES,
            model_id_col=MODEL_ID_COL,
            prediction_col=PREDICTION_COL,
            label_col=LABEL_COL,
            problem_type=PROBLEM_TYPE,
        ),
        output_schema_name=output_schema_name,
        schedule=None,  # We will refresh the metrics on-demand in this notebook
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS,
        custom_metrics=CUSTOM_METRICS
    )
    print(info)
except Exception as e:
    # Ensure the exception was expected
    assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    lm.update_monitor(
        table_name=processed_requests_table_name,
        updated_params=dict(
            profile_type=lm.InferenceLog(
                timestamp_col=TIMESTAMP_COL,
                granularities=GRANULARITIES,
                model_id_col=MODEL_ID_COL,
                prediction_col=PREDICTION_COL,
                label_col=LABEL_COL,
                problem_type=PROBLEM_TYPE,
            ),
            output_schema_name=output_schema_name,
            schedule=None,
            baseline_table_name=BASELINE_TABLE,
            slicing_exprs=SLICING_EXPRS,
            custom_metrics=CUSTOM_METRICS
        )
    )

    # Refresh metrics calculated on the requests table.
    refresh_info = lm.run_refresh(table_name=processed_requests_table_name)
    print(refresh_info)

# COMMAND ----------

# MAGIC %md ### Optimize tables
# MAGIC
# MAGIC For optimal performance, tables must be `OPTIMIZED` regularly. Compaction is performed once a day (at most).

# COMMAND ----------

# We do not VACUUM the final table being monitored because
# Lakehouse Monitoring depends on the Change Data Feed to optimize its computation.
for table_name, should_vacuum in (
    (payload_table_name, True),
    (unpacked_requests_table_name, True),
    (processed_requests_table_name, False),
):
    optimize_table_daily(delta_table=DeltaTable.forName(spark, table_name), vacuum=should_vacuum)
