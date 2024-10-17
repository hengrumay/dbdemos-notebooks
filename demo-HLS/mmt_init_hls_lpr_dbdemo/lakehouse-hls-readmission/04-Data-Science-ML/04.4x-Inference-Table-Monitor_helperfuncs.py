# Databricks notebook source
# MAGIC %md
# MAGIC ## Helperfuncs for Inference Table Monitoring   
# MAGIC - some code updates with comments `-- updated mmt 2024Oct`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC Install the Python SDK and restart Python.

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.28.0"

# COMMAND ----------

# This step is necessary to reset the environment with our newly installed wheel.
dbutils.library.restartPython()

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

    # Fetch the PAT token to send in the API request -- updated mmt 2024Oct
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


def delimit_identifier(identifier: str, special_char_compatible: bool = False) -> str:
    """
    Delimits an identifier name using backticks to handle special characters.
    For example, "endpoint-inference-table" becomes `endpoint-inference-table`.

    :param identifier: Name of the identifier to delimit
    :return: Delimited identifier name
    """
    # return f"`{identifier.replace('`', '``')}`"

    ## Include special_char_compatible bool as additional args | -- updated mmt 2024Oct
    if special_char_compatible == False:
        return f"{identifier.replace('`', '``')}" #removed backticks "`" -- updated mmt 2024Oct
    else:
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

# DBTITLE 1,previous version
# """
# Conversion helper functions.
# """
# def convert_to_record_json(json_str: str) -> str:
#     """
#     Converts records from the four accepted JSON formats for Databricks
#     Model Serving endpoints into a common, record-oriented
#     DataFrame format which can be parsed by the PySpark function from_json.
    
#     :param json_str: The JSON string containing the request or response payload.
#     :return: A JSON string containing the converted payload in record-oriented format.
#     """
#     try:
#         request = json.loads(json_str)
#     except json.JSONDecodeError:
#         return json_str
#     output = []
#     if isinstance(request, dict):
#         obj_keys = set(request.keys())
#         if "dataframe_records" in obj_keys:
#             # Record-oriented DataFrame
#             output.extend(request["dataframe_records"])
#         elif "dataframe_split" in obj_keys:
#             # Split-oriented DataFrame
#             dataframe_split = request["dataframe_split"]
#             output.extend([dict(zip(dataframe_split["columns"], values)) for values in dataframe_split["data"]])
#         elif "instances" in obj_keys:
#             # TF serving instances
#             output.extend(request["instances"])
#         elif "inputs" in obj_keys:
#             # TF serving inputs
#             output.extend([dict(zip(request["inputs"], values)) for values in zip(*request["inputs"].values())])
#         elif "predictions" in obj_keys:
#             # Predictions
#             output.extend([{PREDICTION_COL: prediction} for prediction in request["predictions"]])
#         return json.dumps(output)
#     else:
#         # Unsupported format, pass through
#         return json_str


# @F.pandas_udf(T.StringType())
# def json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
#     """A UDF to apply the JSON conversion function to every request/response."""
#     return json_strs.apply(convert_to_record_json)


# """
# Request helper functions.
# """
# def get_endpoint_status(endpoint_name: str) -> Dict:
#     """
#     Fetches the status and config of and endpoint using the `serving-endpoints` REST endpoint.
    
#     :param endpoint_name: Name of the serving endpoint
#     :return: Dict containing JSON status response
    
#     """
#     # Fetch the PAT token to send in the API request
#     workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
#     token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

#     url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
#     headers = {"Authorization": f"Bearer {token}"}
#     request = dict(name=endpoint_name)
#     response = requests.get(url, json=request, headers=headers)
    
#     # Check for unauthorization errors due to PAT token
#     if "unauthorized" in response.text.lower():
#         raise Exception(
#           f"Unable to retrieve status for endpoint '{endpoint_name}'. "
#           "If you are an admin, please try using a cluster in Single User security mode."
#         )
        
#     response_json = response.json()

#     # Verify that Model Serving is enabled.
#     if "state" not in response_json:
#         raise Exception(f"Model Serving is not enabled for endpoint {endpoint_name}. "
#                         "Please validate the status of this endpoint before running this notebook.")

#     # Verify that Inference Tables is enabled.
#     if "auto_capture_config" not in response_json["config"] or not response_json["config"]["auto_capture_config"]["enabled"]:
#         raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. "
#                         "Please create an endpoint with Inference Tables enabled before running this notebook.")

#     return response_json


# def delimit_identifier(identifier: str) -> str:
#     """
#     Delimits an identifier name using backticks to handle special characters.
#     For example, "endpoint-inference-table" becomes `endpoint-inference-table`.

#     :param identifier: Name of the identifier to delimit
#     :return: Delimited identifier name
#     """
#     return f"`{identifier.replace('`', '``')}`"


# @dataclasses.dataclass
# class InferenceTableConfig:
#     """
#     Data class to store configuration info for Inference Tables.
#     """
#     # Delimited name of the catalog.
#     catalog: str
#     # Delimited name of the schema.
#     schema: str
#     # Delimited and fully qualified name of the payload table.
#     payload_table_name: str
#     # Delimited and fully qualified name of the unpacked requests table.
#     unpacked_requests_table_name: str
#     # Delimited and fully qualified name of the processed requests table.
#     processed_requests_table_name: str


# def collect_inference_table_config(endpoint_name: str) -> InferenceTableConfig:
#     """
#     Collects the Inference Table configuration necessary to reference all tables used by this notebook:
#         - The payload table configured by the endpoint
#         - The unpacked requests table produced by this notebook
#         - The processed requests table produced by this notebook
    
#     Note that this relies on fetching a Personal Access Token (PAT) from the
#     runtime context, which can fail in certain scenarios if run by an admin on a shared cluster.
#     If you are experiencing issues, you can try switch to a Single User cluster.
        
#     :param endpoint_name: Name of the serving endpoint
#     :return: InferenceTableConfig containing Unity Catalog identifiers
#     """
#     response_json = get_endpoint_status(endpoint_name=endpoint_name)

#     auto_capture_config = response_json["config"]["auto_capture_config"]
#     catalog = auto_capture_config["catalog_name"]
#     schema = auto_capture_config["schema_name"]
#     table_name_prefix = auto_capture_config["table_name_prefix"]

#     # These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
#     payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
#     unpacked_table_name = f"{table_name_prefix}_unpacked"
#     processed_table_name = f"{table_name_prefix}_processed"

#     # Escape identifiers with backticks before returning to handle special characters
#     delimited_catalog = delimit_identifier(catalog)
#     delimited_schema = delimit_identifier(schema)
#     delimited_qualified_schema = f"{delimited_catalog}.{delimited_schema}"
#     return InferenceTableConfig(
#         catalog=delimited_catalog,
#         schema=delimited_schema,
#         payload_table_name=f"{delimited_qualified_schema}.{delimit_identifier(payload_table_name)}",
#         unpacked_requests_table_name=f"{delimited_qualified_schema}.{delimit_identifier(unpacked_table_name)}",
#         processed_requests_table_name=f"{delimited_qualified_schema}.{delimit_identifier(processed_table_name)}",
#     )


# def get_served_models(endpoint_name: str) -> List[Dict]:
#     """
#     Fetches the list of models being served by an endpoint.
    
#     :param endpoint_name: Name of the serving endpoint
#     :return: List of Dicts for each served model
#     """
#     response_json = get_endpoint_status(endpoint_name=endpoint_name)
#     if "config" not in response_json and "pending_config" not in response_json:
#         raise Exception(f"Unable to find any config for endpoint '{endpoint_name}'.")
#     config = response_json["config"] if "config" in response_json else response_json["pending_config"]
#     if "served_models" not in config or len(config["served_models"]) == 0:
#         raise Exception(f"Unable to find any served models for endpoint '{endpoint_name}'.")
    
#     served_models = config["served_models"]
#     return served_models


# def convert_numpy_dtype_to_spark(dtype: np.dtype) -> T.DataType:
#     """
#     Converts the input numpy type to a Spark type.
    
#     :param dtype: The numpy type
#     :return: The Spark data type
#     """
#     NUMPY_SPARK_DATATYPE_MAPPING = {
#         np.byte: T.LongType(),
#         np.short: T.LongType(),
#         np.intc: T.LongType(),
#         np.int_: T.LongType(),
#         np.longlong: T.LongType(),
#         np.ubyte: T.LongType(),
#         np.ushort: T.LongType(),
#         np.uintc: T.LongType(),
#         np.half: T.DoubleType(),
#         np.single: T.DoubleType(),
#         np.float_: T.DoubleType(),
#         np.bool_: T.BooleanType(),
#         np.object_: T.StringType(),
#         np.str_: T.StringType(),
#         np.unicode_: T.StringType(),
#         np.bytes_: T.BinaryType(),
#         np.timedelta64: T.LongType(),
#         np.datetime64: T.TimestampType(),
#     }
#     for source, target in NUMPY_SPARK_DATATYPE_MAPPING.items():
#         if np.issubdtype(dtype, source):
#             return target
#     raise ValueError(f"Unsupported numpy dtype: {dtype}")


# def infer_request_response_fields(endpoint_name: str) -> Tuple[Optional[List[T.StructField]], Optional[T.StructField]]:
#     """
#     Infers the request and response schema of the endpoint by loading the signature
#     of the first available served model with a logged signature
#     and extracting the Spark struct fields for each respective schema.
    
#     Note that if the signature varies across served models within the endpoint, this will only
#     use the first available one; if you need to handle multiple signatures, please use the
#     REQUEST_FIELDS and RESPONSE_FIELD parameters at the top of the notebook.
    
#     Raises an error if:
#     - The endpoint doesn't exist
#     - The endpoint doesn't have a served model with a logged signature
    
#     :param endpoint_name: Name of the serving endpoint to infer schemas for
#     :return: A tuple containing a list of struct fields for the request schema, and a
#              single struct field for the response. Either element may be None if this
#              endpoint's models' signatures did not contain an input or output signature, respectively.
#     """
#     # Load the first model (with a logged signature) being served by this endpoint
#     served_models = get_served_models(endpoint_name=endpoint_name)
#     signature = None
#     for served_model in served_models:
#         model_name = served_model["model_name"]
#         model_version = served_model["model_version"]
#         loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
#         if loaded_model.metadata.signature is not None:
#             signature = loaded_model.metadata.signature
#             break

#     if signature is None:
#         raise Exception("One of REQUEST_FIELDS or RESPONSE_FIELD was not specified, "
#                         "but endpoint has no served models with a logged signature. Please define the schemas "
#                         "in the Parameters section of this notebook.")
    
#     # Infer the request schema from the model signature
#     request_fields = None if signature.inputs is None else signature.inputs.as_spark_schema().fields
#     if signature.outputs is None:
#         response_field = None
#     else:
#         # Get the Spark datatype for the model output
#         model_output_schema = signature.outputs
#         if model_output_schema.is_tensor_spec():
#             if len(model_output_schema.input_types()) > 1:
#                 raise ValueError("Models with multiple outputs are not supported for monitoring")
#             output_type = convert_numpy_dtype_to_spark(model_output_schema.numpy_types()[0])
#         else:
#             output_type = model_output_schema.as_spark_schema()
#             if isinstance(output_type, T.StructType):
#                 if len(output_type.fields) > 1:
#                     raise ValueError(
#                         "Models with multiple outputs are not supported for monitoring")
#                 else:
#                     output_type = output_type[0].dataType
#         response_field = T.StructField(PREDICTION_COL, output_type)
        
#     return request_fields, response_field
    

# def process_requests(requests_raw: DataFrame, request_fields: List[T.StructField], response_field: T.StructField) -> DataFrame:
#     """
#     Takes a stream of raw requests and processes them by:
#         - Unpacking JSON payloads for requests and responses
#         - Exploding batched requests into individual rows
#         - Converting Unix epoch millisecond timestamps to be Spark TimestampType
        
#     :param requests_raw: DataFrame containing raw requests. Assumed to contain the following columns:
#                             - `request`
#                             - `response`
#                             - `timestamp_ms`
#     :param request_fields: List of StructFields representing the request schema
#     :param response_field: A StructField representing the response schema
#     :return: A DataFrame containing processed requests
#     """
#     # Convert the timestamp milliseconds to TimestampType for downstream processing.
#     requests_timestamped = requests_raw \
#         .withColumn(TIMESTAMP_COL, (F.col("timestamp_ms") / 1000).cast(T.TimestampType())) \
#         .drop("timestamp_ms")

#     # Convert the model name and version columns into a model identifier column.
#     requests_identified = requests_timestamped \
#         .withColumn(MODEL_ID_COL, F.concat(F.col("request_metadata").getItem("model_name"), F.lit("_"), F.col("request_metadata").getItem("model_version"))) \
#         .drop("request_metadata")

#     # Rename the date column to avoid collisions with features.
#     requests_dated = requests_identified.withColumnRenamed("date", DATE_COL)

#     # Consolidate and unpack JSON.
#     request_schema = T.ArrayType(T.StructType(request_fields))
#     response_schema = T.ArrayType(T.StructType([response_field]))
#     requests_unpacked = requests_dated \
#         .withColumn("request", json_consolidation_udf(F.col("request"))) \
#         .withColumn("response", json_consolidation_udf(F.col("response"))) \
#         .withColumn("request", F.from_json(F.col("request"), request_schema)) \
#         .withColumn("response", F.from_json(F.col("response"), response_schema))

#     # Explode batched requests into individual rows.
#     DB_PREFIX = "__db"
#     requests_exploded = requests_unpacked \
#         .withColumn(f"{DB_PREFIX}_request_response", F.arrays_zip(F.col("request"), F.col("response"))) \
#         .withColumn(f"{DB_PREFIX}_request_response", F.explode(F.col(f"{DB_PREFIX}_request_response"))) \
#         .select(F.col("*"), F.col(f"{DB_PREFIX}_request_response.request.*"), F.col(f"{DB_PREFIX}_request_response.response.*")) \
#         .drop(f"{DB_PREFIX}_request_response", "request", "response")

#     # Generate an example ID so we can de-dup each row later when upserting results.
#     requests_processed = requests_exploded \
#         .withColumn(EXAMPLE_ID_COL, F.expr("uuid()"))
    
#     return requests_processed


# """
# Table helper functions.
# """
# def initialize_table(fully_qualified_table_name: str, schema: T.StructType, special_char_compatible: bool = False) -> None:
#     """
#     Initializes an output table with the Delta Change Data Feed.
#     All tables are partitioned by the date column for optimal performance.
    
#     :param fully_qualified_table_name: Fully qualified name of the table to initialize, like "{catalog}.{schema}.{table}"
#     :param schema: Spark schema of the table to initialize
#     :param special_char_compatible: Boolean to determine whether to upgrade the min reader/writer
#                                     version of Delta and use column name mapping mode. If True,
#                                     this allows for column names with spaces and special characters, but
#                                     it also prevents these tables from being read by external systems
#                                     outside of Databricks. If the latter is a requirement, and the model
#                                     requests contain feature names containing only alphanumeric or underscore
#                                     characters, set this flag to False.
#     :return: None
#     """
#     dt_builder = DeltaTable.createIfNotExists(spark) \
#         .tableName(fully_qualified_table_name) \
#         .addColumns(schema) \
#         .partitionedBy(DATE_COL)
    
#     if special_char_compatible:
#         dt_builder = dt_builder \
#             .property("delta.enableChangeDataFeed", "true") \
#             .property("delta.columnMapping.mode", "name") \
#             .property("delta.minReaderVersion", "2") \
#             .property("delta.minWriterVersion", "5")
    
#     dt_builder.execute()
    
    
# def read_requests_as_stream(fully_qualified_table_name: str) -> DataFrame:
#     """
#     Reads the endpoint's Inference Table to return a single streaming DataFrame
#     for downstream processing that contains all request logs.

#     If the endpoint is not logging data, will raise an Exception.
    
#     :param fully_qualified_table_name: Fully qualified name of the payload table
#     :return: A Streaming DataFrame containing request logs
#     """
#     try:
#         return spark.readStream.format("delta").table(fully_qualified_table_name)
#     except AnalysisException:
#         raise Exception("No payloads have been logged to the provided Inference Table location. "
#                         "Please be sure Inference Tables is enabled and ready before running this notebook.")


# def optimize_table_daily(delta_table: DeltaTable, vacuum: bool = False) -> None:
#     """
#     Runs OPTIMIZE on the provided table if it has not been run already today.
    
#     :param delta_table: DeltaTable object representing the table to optimize.
#     :param vacuum: If true, will VACUUM commit history older than default retention period.
#     :return: None
#     """
#     # Get the full history of the table.
#     history_df = delta_table.history()
    
#     # Filter for OPTIMIZE operations that have happened today.
#     optimize_today = history_df.filter(F.to_date(F.col("timestamp")) == F.current_date())
#     if optimize_today.count() == 0:
#         # No OPTIMIZE has been run today, so initiate it.
#         delta_table.optimize().executeCompaction()
        
#         # Run VACUUM if specified.
#         if vacuum:
#             delta_table.vacuum()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,store token as secrets
# %sh
# databricks secrets create-scope --scope <scope-name>
# databricks secrets put --scope <scope-name> --key <key-name>

# COMMAND ----------

# DBTITLE 1,list  secrets
# dbutils.secrets.list("<scope-name>")
# dbutils.secrets.list("mmt")
## [SecretMetadata(key='databricks_token')] 

# COMMAND ----------

# DBTITLE 1,workspace_url & token
# # Retrieve the Databricks workspace URL from environment variables
# workspace_url = os.environ.get("DATABRICKS_HOST")

# # Retrieve your Databricks secret (PAT) securely
# token = dbutils.secrets.get(scope="mmt", key="databricks_token")

# COMMAND ----------

# endpoint_name = "dbdemos_hls_pr_endpoint"
# workspace_url = "https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485"
# #os.environ.get("DATABRICKS_HOST")
# # Retrieve your Databricks secret (PAT) securely
# token = dbutils.secrets.get(scope="mmt", key="databricks_token")

# url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
# headers = {"Authorization": f"Bearer {token}"}
# request = dict(name=endpoint_name)
# response = requests.get(url, json=request, headers=headers)

# COMMAND ----------

# response
