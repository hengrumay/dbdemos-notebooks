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
# MAGIC   - https://docs.databricks.com/_extras/notebooks/source/monitoring/inference-table-monitor.html    
# MAGIC   this template code is used in this notebook and includes the helperfuncs code in a single notebook which is split up in this walkthrough and run in the Setup section cell just below
# MAGIC - https://docs.databricks.com/en/lakehouse-monitoring/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Install SDK + Initialize HelperFunctions  
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run SDKs + HelperFunctions NB
# MAGIC %run ./04.4x-Inference-Table-Monitor_helperfuncs

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
# MAGIC
# MAGIC https://docs.databricks.com/en/lakehouse-monitoring/monitor-output.html

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
RESPONSE_FIELD = T.StructField('predictions', T.DoubleType(), True) ## must match LABEL_COL schema

# For each table to join with, provide a tuple of: (table name, columns to join/'add', columns to use for equi-join)
# For the equi-join key columns, this will be either "client_request_id" or a combination of features that uniquely identifies an example to be joined with.
# Tables must be registered in **Unity Catalog**
JOIN_TABLES = [ # Example: ("labels_table", ["labels", "client_request_id"], ["client_request_id"])
               (
                "mmt_demos.hls_readmission_dbdemoinit.training_dataset", ## "groundtruth"      
                ["ENCOUNTER_ID","30_DAY_READMISSION"], 
                ["ENCOUNTER_ID"]    
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

# DBTITLE 1,[illustrate] inferring model signature
# inferred_request_fields, inferred_response_field = infer_request_response_fields(endpoint_name=ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,inferred REQUEST
# inferred_request_fields

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
# inferred_response_field

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

payload_table_name = inference_table_config.payload_table_name ## 
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

# display(requests_raw)

# COMMAND ----------

# display(requests_unpacked)

# COMMAND ----------

# MAGIC %md ### Join with labels/data
# MAGIC
# MAGIC In this section, we optionally join the requests with other tables that contain relevant data, such as inference labels or other business metrics.</br>Note that this cell should be run even if there are no tables yet to join with.
# MAGIC
# MAGIC In order to persist the join, we do an `upsert`: insert new rows if they don't yet exist, or update them if they do.</br>This allows us to attach late-arriving information to requests.

# COMMAND ----------

# DBTITLE 0,reset/del request_cleaned/processed
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
# display(requests_cleaned.groupby("30_DAY_READMISSION").agg(F.count('predictions')))
# display(requests_cleaned.groupby("predictions").agg(F.count('predictions')))
# display(requests_cleaned.groupby("30_DAY_READMISSION","predictions").agg(F.count('predictions')).sort('30_DAY_READMISSION') )

# COMMAND ----------

# DBTITLE 1,check flattened+joined inference_[v#]_processed
print(processed_requests_table_name)
testdf = spark.table(processed_requests_table_name)

# COMMAND ----------

# DBTITLE 1,Change Data Feed
# https://docs.databricks.com/en/delta/delta-change-data-feed.html#enable-change-data-feed
# [NOT YET IMPLEMENTED; when doing so -- do it once ] 
# ALTER TABLE myDeltaTable SET TBLPROPERTIES (delta.enableChangeDataFeed = true) 

# COMMAND ----------

# MAGIC %md ### Monitor the inference table
# MAGIC
# MAGIC In this step, we create a monitor on our inference table by using the `create_monitor` API. If the monitor already exists, we pass the same parameters to `update_monitor`. In steady state, this should result in no change to the monitor.
# MAGIC
# MAGIC Afterwards, we queue a metric refresh so that the monitor analyzes the latest processed requests.
# MAGIC
# MAGIC See the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/index.html)|[Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/index)) for more details on the parameters and the expected usage.
# MAGIC
# MAGIC ref: https://docs.databricks.com/en/lakehouse-monitoring/create-monitor-api.html
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()

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
        schedule=None,  # We will refresh the profile/drift metrics on-demand in this notebook ##
        # schedule=MonitorCronSchedule(quartz_cron_expression="0 0 12 * * ?",  # schedules a refresh every day at 12 noon
        #                              timezone_id="PST"
        #                             ),
        # notifications=MonitorNotifications(
        #                                     on_failure=MonitorDestination(email_addresses=["your_email@domain.com"])
        #                                   ),
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS,
        custom_metrics=CUSTOM_METRICS,
        assets_dir=assets_dir
    )
    print(info)
except Exception as e:
    # Ensure the exception was expected
    # assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"    
    assert "already exists" in str(e), f"Unexpected error: {e}" ## mmt updated 2024Oct
    
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
        schedule=None, # We will refresh the profile/drift metrics on-demand in this notebook ##
        # schedule=MonitorCronSchedule(quartz_cron_expression="0 0 12 * * ?",  # schedules a refresh every day at 12 noon
        #                              timezone_id="PST"
        #                             ),
        # notifications=MonitorNotifications(
        #                                     on_failure=MonitorDestination(email_addresses=["your_email@domain.com"])
        #                                   ),
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS,
        custom_metrics=CUSTOM_METRICS,
    )

    # Refresh metrics calculated on the requests table.
    refresh_info = w.quality_monitors.run_refresh(table_name=processed_requests_table_name)
    print(refresh_info)

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

# DBTITLE 1,Monitoring API ref
# https://api-docs.databricks.com/python/lakehouse-monitoring/latest/databricks.lakehouse_monitoring.html
# https://docs.databricks.com/en/lakehouse-monitoring/custom-metrics.html

# COMMAND ----------

# DBTITLE 1,Example Monitoring API calls
# w = WorkspaceClient()

## Refresh Metrics: To refresh the metrics tables, use the run_refresh method. This can be done manually or scheduled as shown in the example above. 

# w.quality_monitors.run_refresh(table_name=f"{catalog}.{schema}.{table_name}")


## View and Manage Monitors: You can list, get the status of, and cancel refreshes using the following methods:12

# # List refreshes
# w.quality_monitors.list_refreshes(
#     table_name=f"{catalog}.{schema}.{table_name}"
# )

# # Get the status of a specific refresh
# run_info = w.quality_monitors.run_refresh(
#     table_name=f"{catalog}.{schema}.{table_name}"
# )
# w.quality_monitors.get_refresh(
#     table_name=f"{catalog}.{schema}.{table_name}",
#     refresh_id=run_info.refresh_id
# )

# # Cancel a refresh
# w.quality_monitors.cancel_refresh(
#     table_name=f"{catalog}.{schema}.{table_name}",
#     refresh_id=run_info.refresh_id
# )

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

