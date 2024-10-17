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
# MAGIC
# MAGIC ---    
# MAGIC Refs: 
# MAGIC - https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html
# MAGIC - https://docs.databricks.com/en/lakehouse-monitoring/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Install SDK + Initialize HelperFunctions  
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run SDKs + HelperFunctions NB
# MAGIC %run ./04.4x-Inference-Table-Monitor_helperfuncs

# COMMAND ----------

# %pip install "databricks-sdk>=0.28.0"

# COMMAND ----------

# This step is necessary to reset the environment with our newly installed wheel.
# dbutils.library.restartPython()

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

# DBTITLE 1,Monitor PARAMETERS**
from pyspark.sql import functions as F, types as T

"""
Required parameters in order to run this notebook.
"""
ENDPOINT_NAME = 'dbdemos_hls_pr_endpoint_v2'  ## 2update     # Name of the model serving endpoint
PROBLEM_TYPE = 'classification'                              # ML problem type, one of "classification"/"regression"

# Validate that all required inputs have been provided
if None in [ENDPOINT_NAME, PROBLEM_TYPE]:
    raise Exception("Please fill in the required information for endpoint name and problem type.")

"""
Optional parameters to control monitoring analysis.
"""
# Monitoring configuration parameters. For help, use the command help(w.quality_monitors.create).
GRANULARITIES = ["1 hour", "1 day"] #, "1 week", "1 month", "1 year"]              # Window sizes to analyze data over

# SLICING_EXPRS =  None                          # Expressions to slice data with
SLICING_EXPRS =  ["age_at_encounter < 18", 
                  "age_at_encounter >= 18 AND age_at_encounter < 60", 
                  "age_at_encounter >= 60",
                  # "ETHNICITY_hispanic", "ETHNICITY_nonhispanic", "GENDER_F=1", "GENDER_M=1", 
                  # "INCOME", "BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE", "enc_length", 
                  # "ENCOUNTERCLASS_emergency=1", "ENCOUNTERCLASS_ambulatory=1", "ENCOUNTERCLASS_inpatient=1", "ENCOUNTERCLASS_outpatient=1", "ENCOUNTERCLASS_wellness=1",
                  ]                   
CUSTOM_METRICS = None                            # A list of custom metrics to compute
BASELINE_TABLE = None                            # Baseline table name, if any, for computing baseline drift

"""
Optional parameters to control processed tables.
"""
PREDICTION_COL = "predictions"                   # What to name the prediction column in the processed table
LABEL_COL = "30_DAY_READMISSION"  #None          # Name of columns holding labels. Change this if you join your own labels via JOIN_TABLES above.
## LABEL_COL if used will require the same schema as PREDICTION_COL

# The endpoint input schema, e.g.: [T.StructField("feature_one", T.T.StringType()), T.StructField("feature_two", T.IntegerType())]
## If not provided, will attempt to infer request fields from one of the logged models' signatures.
# REQUEST_FIELDS = None
REQUEST_FIELDS = [T.StructField('MARITAL_M', T.LongType(), False),
                  T.StructField('MARITAL_S', T.LongType(), False),
                  T.StructField('RACE_asian', T.LongType(), False),
                  T.StructField('RACE_black', T.LongType(), False),
                  T.StructField('RACE_hawaiian', T.LongType(), False),
                  T.StructField('RACE_other', T.LongType(), False),
                  T.StructField('RACE_white', T.LongType(), False),
                  T.StructField('ETHNICITY_hispanic', T.LongType(), False),
                  T.StructField('ETHNICITY_nonhispanic', T.LongType(), False),
                  T.StructField('GENDER_F', T.LongType(), False),
                  T.StructField('GENDER_M', T.LongType(), False),
                  T.StructField('INCOME', T.LongType(), False),
                  T.StructField('BASE_ENCOUNTER_COST', T.DoubleType(), False),
                  T.StructField('TOTAL_CLAIM_COST', T.DoubleType(), False),
                  T.StructField('PAYER_COVERAGE', T.DoubleType(), False),
                  T.StructField('enc_length', T.LongType(), False),
                  T.StructField('ENCOUNTERCLASS_ambulatory', T.LongType(), False),
                  T.StructField('ENCOUNTERCLASS_emergency', T.LongType(), False),
                  T.StructField('ENCOUNTERCLASS_hospice', T.LongType(), False),
                  T.StructField('ENCOUNTERCLASS_inpatient', T.LongType(), False),
                  T.StructField('ENCOUNTERCLASS_outpatient', T.LongType(), False),
                  T.StructField('ENCOUNTERCLASS_wellness', T.LongType(), False),
                  T.StructField('age_at_encounter', T.DoubleType(), False),
                  T.StructField('patient_id', T.StringType(), False), ## not part of model features processing; used for baseline datajoining
                  T.StructField('ENCOUNTER_ID', T.StringType(), False) ## not part of model features processing; used for baseline datajoining
                 ]

# The endpoint output schema, e.g.: T.StructField(PREDICTION_COL, T.IntegerType()). Field name must be PREDICTION_COL.
## If not provided, will attempt to infer response field from one of the logged models' signatures.
# RESPONSE_FIELD = None
RESPONSE_FIELD = T.StructField('predictions', T.DoubleType(), True) ## 2update

# For each table to join with, provide a tuple of: (table name, columns to join/'add', columns to use for equi-join)
# For the equi-join key columns, this will be either "client_request_id" or a combination of features that uniquely identifies an example to be joined with.
# Tables must be registered in **Unity Catalog**
JOIN_TABLES = [ # Example: ("labels_table", ["labels", "client_request_id"], ["client_request_id"])
               (
                "mmt_demos.hls_readmission_dbdemoinit.training_dataset", ## "real_groundtruth"      
                ["ENCOUNTER_ID","30_DAY_READMISSION"], #["patient_id","ENCOUNTER_ID","30_DAY_READMISSION"],
                ["ENCOUNTER_ID"]    ## ["patient_id","ENCOUNTER_ID"], #
               )
              ]

# Optionally filter the request logs based on an expression that can be used in a WHERE clause.
# For example: "timestamp > '2022-10-10' AND age >= 18"
FILTER_EXP = None

# If the request/response schema have special characters (such as spaces) in the column names,
# this flag upgrades the minimum Delta reader/writer version and sets the column mapping mode
# to use names. This in turn reduces compatibility with reading this table on older runtimes
# or by external systems. If your column names have no special characters, set this to False.
SPECIAL_CHAR_COMPATIBLE = False #True

# In order to bound performance of each run, we limit the window of data to be processed from the raw request tables.
# This means that data older than PROCESSING_WINDOW_DAYS will be ignored *if it has not already been processed*.
# If join data (specified by JOIN_TABLES) arrives later than PROCESSING_WINDOW_DAYS, it will be ignored.
# Increase this parameter to ensure you process more data; decrease it to improve performance of this notebook.
PROCESSING_WINDOW_DAYS = 365

# COMMAND ----------

# MAGIC %md ### Imports

# COMMAND ----------

import os ## added

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

# COMMAND ----------

# MAGIC %md ### Helper functions
# MAGIC
# MAGIC Helper functions to read, process, and write the data requests.

# COMMAND ----------

# DBTITLE 1,HelperFunctions
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
            output.extend(request["inputs"])
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
    # Fetch the API token to send in the API request ## 
    # workspace_url = "<YOUR-WORKSPACE-URL>"
    # token = "<YOUR-API-TOKEN>"

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

    # Filter out the non-successful requests.
    requests_success = requests_dated.filter(F.col("status_code") == "200")

    # Consolidate and unpack JSON.
    request_schema = T.ArrayType(T.StructType(request_fields))
    response_schema = T.ArrayType(T.StructType([response_field]))
    requests_unpacked = requests_success \
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

# DBTITLE 1,inferred REQUEST
# REQUEST_FIELDS

# [StructField('MARITAL_M', LongType(), False),
#  StructField('MARITAL_S', LongType(), False),
#  StructField('RACE_asian', LongType(), False),
#  StructField('RACE_black', LongType(), False),
#  StructField('RACE_hawaiian', LongType(), False),
#  StructField('RACE_other', LongType(), False),
#  StructField('RACE_white', LongType(), False),
#  StructField('ETHNICITY_hispanic', LongType(), False),
#  StructField('ETHNICITY_nonhispanic', LongType(), False),
#  StructField('GENDER_F', LongType(), False),
#  StructField('GENDER_M', LongType(), False),
#  StructField('INCOME', LongType(), False),
#  StructField('BASE_ENCOUNTER_COST', DoubleType(), False),
#  StructField('TOTAL_CLAIM_COST', DoubleType(), False),
#  StructField('PAYER_COVERAGE', DoubleType(), False),
#  StructField('enc_length', LongType(), False),
#  StructField('ENCOUNTERCLASS_ambulatory', LongType(), False),
#  StructField('ENCOUNTERCLASS_emergency', LongType(), False),
#  StructField('ENCOUNTERCLASS_hospice', LongType(), False),
#  StructField('ENCOUNTERCLASS_inpatient', LongType(), False),
#  StructField('ENCOUNTERCLASS_outpatient', LongType(), False),
#  StructField('ENCOUNTERCLASS_wellness', LongType(), False),
#  StructField('age_at_encounter', DoubleType(), False)]


# COMMAND ----------

# DBTITLE 1,inferred RESPONSE
# RESPONSE_FIELD
# StructField('predictions', LongType(), True)

# COMMAND ----------

# MAGIC %md ### Process request logs

# COMMAND ----------

# MAGIC %md #### Collect Inference Table configuration
# MAGIC
# MAGIC First, we fetch the configurated Unity Catalog location for the payload table produced by the endpoint.</br></br>**NOTE**: this request may fail if you are a workspace admin and are not using the **Single User** cluster security mode.</br>In such cases, please switch to a Single User cluster.

# COMMAND ----------

## the UC namespace `backticks` formatting needs updating else the downstream monitoring part will fail -- updated in HelperFuncs NB

# COMMAND ----------

# DBTITLE 1,get payload + set unpack/process-ed tables
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

# DBTITLE 1,drop UC Vol if recreating
# spark.sql(f"drop Volume if exists {inference_table_config.catalog}.{inference_table_config.schema}.`payload-logging`;")

# COMMAND ----------

# DBTITLE 1,>> payload-logging in UC Volumes
# CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.<volume-name>;

spark.sql(f"CREATE VOLUME IF NOT EXISTS {inference_table_config.catalog}.{inference_table_config.schema}.`payload-logging`;")

# COMMAND ----------

# DBTITLE 1,process requests
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
# checkpoint_path = f"dbfs:/payload-logging/{ENDPOINT_NAME}/checkpoint" # original using dbfs
checkpoint_path = f"/Volumes/{inference_table_config.catalog}/{inference_table_config.schema}/payload-logging/{ENDPOINT_NAME}/checkpoint" ## updated to use Volumes mmt 2024Oct

## original
# requests_stream = requests_unpacked.writeStream \
#     .trigger(once=True) \
#     .format("delta") \
#     .partitionBy(DATE_COL) \
#     .outputMode("append") \
#     .option("checkpointLocation", checkpoint_path) \
#     .toTable(unpacked_requests_table_name)
    
# requests_stream.awaitTermination()

## updated mmt--2024Oct
# Placeholder for schema adjustment if necessary
# Ensure requests_unpacked does not have duplicate or conflicting fields
# This might involve renaming columns or selecting specific columns

# Enable schema evolution on the write operation
requests_stream = (requests_unpacked.writeStream 
                    .trigger(once=True) 
                    .format("delta") 
                    .partitionBy(DATE_COL) 
                    .outputMode("append") 
                    .option("checkpointLocation", checkpoint_path) 
                    .option("mergeSchema", "true") # Enable schema evolution
                    .toTable(unpacked_requests_table_name)
                    )
    
requests_stream.awaitTermination()


# COMMAND ----------

display(requests_raw)

# COMMAND ----------

display(requests_unpacked)

# COMMAND ----------

# MAGIC %md ### Join with labels/data
# MAGIC
# MAGIC In this section, we optionally join the requests with other tables that contain relevant data, such as inference labels or other business metrics.</br>Note that this cell should be run even if there are no tables yet to join with.
# MAGIC
# MAGIC In order to persist the join, we do an `upsert`: insert new rows if they don't yet exist, or update them if they do.</br>This allows us to attach late-arriving information to requests.

# COMMAND ----------

# DBTITLE 1,reset/del request_cleaned/processed
# del requests_processed, requests_cleaned

# COMMAND ----------

# DBTITLE 1,original
# # Load the unpacked requests and join them with any specified tables.
# # Filter requests older than PROCESSING_WINDOW_DAYS for optimal preformance.
# requests_processed = spark.table(unpacked_requests_table_name) \
#     .filter(f"CAST({DATE_COL} AS DATE) >= current_date() - (INTERVAL {PROCESSING_WINDOW_DAYS} DAYS)")
# for table_name, preserve_cols, join_cols in JOIN_TABLES:
#     join_data = spark.table(table_name)
#     requests_processed = requests_processed.join(join_data.select(preserve_cols), on=join_cols, how="left")

# # Drop columns that we don't need for monitoring analysis.
# requests_cleaned = requests_processed.drop("status_code", "sampling_fraction", "client_request_id", "databricks_request_id")
    
# # Initialize the processed requests table so we can always use Delta merge operations for joining data.
# initialize_table(
#     fully_qualified_table_name=processed_requests_table_name,
#     schema=requests_cleaned.schema,
#     special_char_compatible=SPECIAL_CHAR_COMPATIBLE,
# )

# # Upsert rows into the existing table, identified by their example_id.
# # Use date in the merge condition in order to allow Spark to dynamically prune partitions.
# merge_cols = [DATE_COL, EXAMPLE_ID_COL]
# merge_condition = " AND ".join([f"existing.{col} = updated.{col}" for col in merge_cols])
# processed_requests_delta_table = DeltaTable.forName(spark, processed_requests_table_name)
# processed_requests_delta_table.alias("existing") \
#     .merge(source=requests_cleaned.alias("updated"), condition=merge_condition) \
#     .whenMatchedUpdateAll() \
#     .whenNotMatchedInsertAll() \
#     .execute()

# COMMAND ----------

# DBTITLE 1,modified
# Load the unpacked requests and join them with any specified tables.
# Filter requests older than PROCESSING_WINDOW_DAYS for optimal performance.
requests_processed = spark.table(unpacked_requests_table_name) \
    .filter(f"CAST({DATE_COL} AS DATE) >= current_date() - (INTERVAL {PROCESSING_WINDOW_DAYS} DAYS)")

# Preprocess the source table to eliminate multiple matches
requests_cleaned = requests_processed.dropDuplicates([EXAMPLE_ID_COL])

# in case of multiple matches [based on the pseudo generated requests], we need to eliminate duplicates in the source table
for table_name, preserve_cols, join_cols in JOIN_TABLES:
    join_data = spark.table(table_name).select(preserve_cols).distinct() #
    requests_cleaned = requests_cleaned.join(join_data, on=join_cols, how="left")

# Drop columns that we don't need for monitoring analysis.
requests_cleaned = requests_cleaned.drop("status_code", "sampling_fraction", "client_request_id", "databricks_request_id")

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

# DBTITLE 1,check
# display(requests_processed)

# COMMAND ----------

# DBTITLE 1,check
display(requests_cleaned)

# COMMAND ----------

# DBTITLE 1,check counts/nulls
display(requests_cleaned.groupby("30_DAY_READMISSION").agg(F.count('predictions')))
display(requests_cleaned.groupby("predictions").agg(F.count('predictions')))
display(requests_cleaned.groupby("30_DAY_READMISSION","predictions").agg(F.count('predictions')).sort('30_DAY_READMISSION') )

# COMMAND ----------

# DBTITLE 1,check flattened+joined inference_[v#]_processed
print(processed_requests_table_name)
testdf = spark.table(processed_requests_table_name)

# COMMAND ----------

# https://docs.databricks.com/en/delta/delta-change-data-feed.html#enable-change-data-feed
# [NOT YET IMPLEMENTED] 
# ALTER TABLE myDeltaTable SET TBLPROPERTIES (delta.enableChangeDataFeed = true) 

# COMMAND ----------

# MAGIC %md ### Monitor the inference table
# MAGIC
# MAGIC In this step, we create a monitor on our inference table by using the `create_monitor` API. If the monitor already exists, we pass the same parameters to `update_monitor`. In steady state, this should result in no change to the monitor.
# MAGIC
# MAGIC Afterwards, we queue a metric refresh so that the monitor analyzes the latest processed requests.
# MAGIC
# MAGIC See the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/index.html)|[Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/index)) for more details on the parameters and the expected usage.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()

# COMMAND ----------

# # Check if the table exists
# table_name = "inference_processed"
# catalog_name = "mmt_demos"
# schema_name = "hls_readmission_dbdemoinit"

# try:
#     # Attempt to query the table to see if it exists
#     spark.sql(f"DESCRIBE TABLE {catalog_name}.{schema_name}.{table_name}")
#     print(f"Table {catalog_name}.{schema_name}.{table_name} exists.")
# except Exception as e:
#     print(f"Table {catalog_name}.{schema_name}.{table_name} does not exist. Error: {e}")

# COMMAND ----------

# DBTITLE 1,Set up to Monitor flattened inference_processed
# Use the catalog/schema of the payload table as the output schema.
output_schema_name = f"{inference_table_config.catalog}.{inference_table_config.schema}"
username = spark.sql("SELECT current_user()").first()["current_user()"]
assets_dir = f"/Workspace/Users/{username}/databricks_lakehouse_monitoring/{processed_requests_table_name}" ##

print(output_schema_name, username, assets_dir)

try:
    info = w.quality_monitors.create(
        table_name=processed_requests_table_name,
        inference_log=MonitorInferenceLog(
            timestamp_col=TIMESTAMP_COL,
            granularities=GRANULARITIES,
            model_id_col=MODEL_ID_COL,
            prediction_col=PREDICTION_COL,
            label_col=LABEL_COL,
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION if PROBLEM_TYPE == "classification" else MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
        ),
        output_schema_name=output_schema_name,
        schedule=None,  # We will refresh the metrics on-demand in this notebook ##
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS,
        custom_metrics=CUSTOM_METRICS,
        assets_dir=assets_dir
    )
    print(info)
except Exception as e:
    # Ensure the exception was expected
    # assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    w.quality_monitors.update(
        table_name=processed_requests_table_name,
        inference_log=MonitorInferenceLog(
            timestamp_col=TIMESTAMP_COL,
            granularities=GRANULARITIES,
            model_id_col=MODEL_ID_COL,
            prediction_col=PREDICTION_COL,
            label_col=LABEL_COL,
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION if PROBLEM_TYPE == "classification" else MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
        ),
        output_schema_name=output_schema_name,
        schedule=None, # We will refresh the metrics on-demand in this notebook ##
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS,
        custom_metrics=CUSTOM_METRICS,
    )

    # Refresh metrics calculated on the requests table.
    refresh_info = w.quality_monitors.run_refresh(table_name=processed_requests_table_name)
    print(refresh_info)

# COMMAND ----------

# # Refresh metrics calculated on the requests table.
# refresh_info = w.quality_monitors.run_refresh(table_name=processed_requests_table_name)
# print(refresh_info)

# COMMAND ----------

# refresh_id = "330830619290446"  # Replace with your actual refresh ID
# # Replace these with your actual catalog, schema, and table name
# catalog = "mmt_demos"
# schema = "hls_readmission_dbdemoinit"
# table_name = 'inference_processed'
# full_table_name = f"{catalog}.{schema}.{table_name}"

# cancel_info = w.quality_monitors.cancel_refresh(
#     table_name=full_table_name,
#     refresh_id=refresh_id
# )
# # display(cancel_info)

# COMMAND ----------

# DBTITLE 1,Refresh Minitor
# try:
#     info = w.quality_monitors.create(
#         table_name=processed_requests_table_name,
#         inference_log=MonitorInferenceLog(
#             timestamp_col=TIMESTAMP_COL,
#             granularities=GRANULARITIES,
#             model_id_col=MODEL_ID_COL,
#             prediction_col=PREDICTION_COL,
#             label_col=LABEL_COL,
#             problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION if PROBLEM_TYPE == "classification" else MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
#         ),
#         output_schema_name=output_schema_name,
#         schedule=None,  # We will refresh the metrics on-demand in this notebook
#         baseline_table_name=BASELINE_TABLE,
#         slicing_exprs=SLICING_EXPRS,
#         custom_metrics=CUSTOM_METRICS,
#         assets_dir=assets_dir
#     )
#     print(info)
# except Exception as e:
#     # Correctly identify the error message for an existing data monitor
#     assert "already exists" in str(e), f"Unexpected error: {e}"

#     # Update the monitor if any parameters of this notebook have changed.
#     w.quality_monitors.update(
#         table_name=processed_requests_table_name,
#         inference_log=MonitorInferenceLog(
#             timestamp_col=TIMESTAMP_COL,
#             # granularities=GRANULARITIES,
#             model_id_col=MODEL_ID_COL,
#             prediction_col=PREDICTION_COL,
#             label_col=LABEL_COL,
#             problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION if PROBLEM_TYPE == "classification" else MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
#         ),
#         output_schema_name=output_schema_name,
#         schedule=None,
#         baseline_table_name=BASELINE_TABLE,
#         slicing_exprs=SLICING_EXPRS,
#         custom_metrics=CUSTOM_METRICS,
#     )

#     # Refresh metrics calculated on the requests table.
#     refresh_info = w.quality_monitors.run_refresh(table_name=processed_requests_table_name)
#     print(refresh_info)

# COMMAND ----------

# MAGIC %md ### Optimize tables
# MAGIC
# MAGIC For optimal performance, tables must be `OPTIMIZED` regularly. Compaction is performed once a day (at most).

# COMMAND ----------

# We **do not** VACUUM the final table being monitored because
# Lakehouse Monitoring depends on the Change Data Feed to optimize its computation.

# ##### NOT RUNNING #####
# for table_name, should_vacuum in (
#     (payload_table_name, True),
#     (unpacked_requests_table_name, True),
#     (processed_requests_table_name, False),
# ):
#     optimize_table_daily(delta_table=DeltaTable.forName(spark, table_name), vacuum=should_vacuum)

# COMMAND ----------

# DBTITLE 1,[optional] deleting existing monitor
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

# w = WorkspaceClient()

# catalog, db, model_name = "mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr".split('.')
# serving_endpoint_name = "dbdemos_hls_pr_endpoint"

# w.quality_monitors.delete(table_name=f"{catalog}.{db}.inference_processed")

# COMMAND ----------

# DBTITLE 1,API ref
# https://api-docs.databricks.com/python/lakehouse-monitoring/latest/databricks.lakehouse_monitoring.html

# COMMAND ----------

# The monitor refresh schedule does include updating the payload processing. When you set up a monitor to run on a scheduled basis, it processes new data according to the profile type you have selected (TimeSeries, InferenceLog, or Snapshot). For TimeSeries and Inference profiles, enabling change data feed (CDF) ensures that only newly appended data is processed, making the execution more efficient. For Snapshot profiles, the entire table is processed with every refresh. You can set up the schedule using the `schedule` parameter in the API or by selecting the frequency and time in the Databricks UI.

# COMMAND ----------

# DBTITLE 1,NOTES
# When a monitor is refreshed, it does not automatically refresh or reprocess the data in the inference_processed table. The refresh operation of a monitor typically involves updating the metrics or visualizations based on the existing data in the inference_processed table or other relevant tables. If there are new data or changes in the inference_processed table, those need to be processed and updated through the pipeline or process that feeds data into this table. The monitor's refresh operation itself does not trigger reprocessing of data; it only updates the monitoring outputs based on the current state of the data it is designed to analyze.


# When scheduling a job to monitor model performance using inference tables, the approach to handling existing data in the inference_processed table, as well as the profile and drift metrics tables, depends on your specific requirements for monitoring and analysis. Here are some considerations and strategies:

# 1. Appending vs. Overwriting Data
# Appending: If you want to maintain a historical record of inference data and metrics for trend analysis over time, you should append new data to the existing tables. This allows you to track changes and potentially identify patterns or issues that develop gradually.

# Overwriting: In some cases, you might only be interested in the most recent data for monitoring purposes. Overwriting the existing data can be suitable for such scenarios, especially if storage space is a concern or if the relevance of data diminishes rapidly over time.

# 2. Handling inference_processed Table
# For the inference_processed table, which contains processed requests and responses, appending new data is typically the preferred approach. This ensures that you have a comprehensive dataset that includes all inference requests and responses over time, which is valuable for detailed analysis and debugging.
# 3. Handling Profile and Drift Metrics Tables
# Profile Metrics Table: Since this table contains summary statistics, you might choose to append new metrics to maintain a history of how these statistics change over time. This can be useful for identifying trends or shifts in the data distribution.

# Drift Metrics Table: For drift metrics, appending new data allows you to track how the distribution of your data changes over time relative to previous windows or a baseline. This is crucial for understanding the stability of your model's performance and identifying when retraining might be necessary.

# 4. Scheduling Considerations
# When scheduling the monitoring job, consider the frequency that aligns with your monitoring needs and the volume of inference data. For instance, more frequent monitoring might be necessary for high-throughput systems or when models are critical to business operations.

# Implement logic in your scheduled job to handle the append or overwrite behavior based on your strategy. For example, you might include checks to avoid duplicating data if appending or to manage the retention of historical data if overwriting.

# 5. Best Practices
# Regularly review the size and growth of your tables, especially if appending data, to ensure that performance remains optimal. Consider implementing data retention policies or archiving older data if necessary.

# Utilize the capabilities of Delta Lake, such as time travel, to manage and query historical data efficiently if you're using Databricks.

# In summary, the decision to append or overwrite data in the inference_processed table and the profile and drift metrics tables depends on your monitoring objectives, storage considerations, and the need for historical analysis. Implementing a thoughtful strategy that aligns with your goals will enable effective and efficient model monitoring.


# When a monitor is refreshed, it does not automatically refresh or reprocess the data in the inference_processed table. The refresh operation of a monitor typically involves updating the metrics or visualizations based on the existing data in the inference_processed table or other relevant tables. If there are new data or changes in the inference_processed table, those need to be processed and updated through the pipeline or process that feeds data into this table. The monitor's refresh operation itself does not trigger reprocessing of data; it only updates the monitoring outputs based on the current state of the data it is designed to analyze..


# To update drift and profile monitor tables in Databricks, you typically need to rerun the monitoring process that generates these tables. This process involves executing the monitoring configuration that specifies the metrics, granularities, slicing expressions, and potentially a baseline table for drift calculations. The specific steps to update these tables depend on how your monitoring setup was initially configured. However, a general approach to update or refresh these tables would involve:

# Reviewing the Current Monitoring Configuration: Ensure that the current configuration (such as granularities, slicing expressions, and baseline table) still meets your monitoring needs. If you need to adjust any parameters, update your monitoring configuration accordingly.

# Rerunning the Monitor: Execute the monitoring process with the updated configuration. This can be done by triggering the Databricks job or notebook that contains your monitoring logic. If you're using Databricks Lakehouse Monitoring, this might involve rerunning the SQL commands or notebooks that create or update the monitor.

# Verifying the Update: After rerunning the monitor, check the drift and profile metric tables to ensure they have been updated as expected. You can query these tables to see the latest metrics and verify that the updates reflect the most recent data and analysis.

# If you're using Databricks Lakehouse Monitoring, the specific commands or steps might vary based on your setup. However, the general approach remains the same: adjust your monitoring configuration if necessary, rerun the monitoring process, and verify the updates.

# Please note that the drift metrics table is only generated if a baseline table is provided, or if a consecutive time window exists after aggregation according to the specified granularities. The profile metrics table contains summary statistics for each column and for each combination of time window, slice, and grouping columns. For InferenceLog analysis, the analysis table also contains model accuracy metrics.

# For more detailed instructions tailored to your specific setup, refer to the Databricks documentation or the configuration of your existing monitoring setup.

# ["Monitor metric tables: Drift and Profile Metrics"](https://docs.databricks.com/en/lakehouse-monitoring/monitor-output.html)

