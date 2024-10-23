# Databricks notebook source
# MAGIC %md
# MAGIC # Monitor Model using Lakehouse Monitoring
# MAGIC In this step, we will leverage Databricks Lakehouse Monitoring([AWS](https://docs.databricks.com/en/lakehouse-monitoring/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/)) to monitor our inference table.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-7.png?raw=true" width="1200">
# MAGIC
# MAGIC Databricks Lakehouse Monitoring attaches a data monitor to any Delta table and it will generate the necessary pipelines to profile the data and calculate quality metrics. You just need to tell it how frequently these quality metrics need to be collected.
# MAGIC
# MAGIC Use Databricks Lakehouse Monitoring to monitor for data drifts, as well as label drift, prediction drift and changes in model quality metrics in Machine Learning use cases. Databricks Lakehouse Monitoring enables monitoring for statistics (e.g. data profiles) and drifts on tables containing:
# MAGIC * batch scoring inferences
# MAGIC * request logs from Model Serving endpoint ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html) |[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/inference-tables))
# MAGIC
# MAGIC Databricks Lakehouse Monitoring stores the data quality and drift metrics in two tables that it automatically creates for each monitored table:
# MAGIC - Profile metrics table (with a `_profile_metrics` suffix)
# MAGIC   - Metrics like percentage of null values, descriptive statistics, model metrics such as accuracy, RMSE, fairness and bias metrics etc.
# MAGIC - Drift metrics table (with a `_drift_metrics` suffix)
# MAGIC   - Metrics like the "delta" between percentage of null values, averages, as well as metrics from statistical tests to detect data drift.
# MAGIC
# MAGIC For demo simplicity purpose, we will use the batch scoring model inference as our inference table. We will attach a monitor to the table `mlops_churn_advanced_inference_table`.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install latest databricks-sdk package (>=0.28.0)
# MAGIC %pip install -qU "databricks-sdk>=0.28.0"
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create monitor
# MAGIC Now, we will create a monitor on top of the inference table. 
# MAGIC It is a one-time setup.

# COMMAND ----------

# MAGIC %md
# MAGIC ### !!!! Check Inference Table
# MAGIC
# MAGIC This can serve as a union for offline & online processed inference.
# MAGIC For simplicity of this demo, we will create the inference table as a copy of the first offline batch prediction table.
# MAGIC
# MAGIC In a different scenario, we could have processed the online inference table and store them in the inference table alongside with the offline inference table.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from patient_readmission_prediction

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from patient_readmission_prediction limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT training_dataset.*, CAST(patient_readmission_prediction.risk_prediction AS DOUBLE) AS predictions, 'mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr_1' AS __db_model_id, inference_v2_processed.__db_timestamp AS __db_timestamp
# MAGIC FROM training_dataset
# MAGIC JOIN patient_readmission_prediction
# MAGIC ON training_dataset.patient_id = patient_readmission_prediction.patient_id
# MAGIC AND training_dataset.encounter_id = patient_readmission_prediction.encounter_id
# MAGIC JOIN inference_v2_processed
# MAGIC ON training_dataset.patient_id = inference_v2_processed.patient_id
# MAGIC AND training_dataset.encounter_id = inference_v2_processed.encounter_id
# MAGIC LIMIT 10

# COMMAND ----------

## add comments!

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE hls_readmission_batch_inference AS
# MAGIC   (SELECT * FROM patient_readmission_prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create baseline table
# MAGIC
# MAGIC For simplification purposes, we will create the baseline table from the pre-existing `risk_prediction` table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT training_dataset.*, patient_readmission_prediction.risk_prediction
# MAGIC FROM training_dataset
# MAGIC JOIN patient_readmission_prediction
# MAGIC ON training_dataset.patient_id = patient_readmission_prediction.patient_id
# MAGIC AND training_dataset.encounter_id = patient_readmission_prediction.encounter_id
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE hls_readmission_baseline AS
# MAGIC   (SELECT training_dataset.*, CAST(patient_readmission_prediction.risk_prediction AS DOUBLE) AS predictions, 'mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr_1' AS __db_model_id
# MAGIC   FROM training_dataset
# MAGIC   JOIN patient_readmission_prediction
# MAGIC   ON training_dataset.patient_id = patient_readmission_prediction.patient_id
# MAGIC   AND training_dataset.encounter_id = patient_readmission_prediction.encounter_id LIMIT 1000)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a custom metric
# MAGIC
# MAGIC In this example, we will calculate the business impact (false predictions in total) of a bad model performance

# COMMAND ----------

from pyspark.sql.types import LongType, StructField
from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType


custom_metric = [
  MonitorMetric(
    type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
    name="positive_prediction_count",
    input_columns=[":table"],
    definition="""sum({{prediction_col}})""",
    output_data_type= StructField("output", LongType()).json()
  )
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create monitor
# MAGIC
# MAGIC As we are monitoring an inference table (including machine learning model predcitions data), we will pick an [Inference profile](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/create-monitor-api#inferencelog-profile) for the monitor.
# MAGIC

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType


print(f"Creating monitor for inference table {catalog}.{db}.hls_readmission_batch_inference")
w = WorkspaceClient()

info = w.quality_monitors.create(
  table_name=f"{catalog}.{db}.hls_readmission_batch_inference",
  inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="predictions",
        timestamp_col="__db_timestamp",
        granularities=["1 day"],
        model_id_col="__db_model_id",
        label_col="30_DAY_READMISSION", # optional
  ),
  assets_dir=f"{os.getcwd()}/monitoring", # Change this to another folder of choice if needed
  output_schema_name=f"{catalog}.{db}",
  baseline_table_name=f"{catalog}.{db}.hls_readmission_baseline",
  slicing_exprs=None, #["senior_citizen='Yes'", "contract"], # Slicing dimension
  custom_metrics=custom_metric
)

# COMMAND ----------

# MAGIC %md Wait/Verify that monitor was created

# COMMAND ----------

import time
from databricks.sdk.service.catalog import MonitorInfoStatus, MonitorRefreshInfoState


# Wait for monitor to be created
while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.get(table_name=f"{catalog}.{db}.hls_readmission_batch_inference")
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"

# COMMAND ----------

# MAGIC %md Monitor creation for the first time will also **trigger an initial refresh** so fetch/wait or trigger a monitoring job and wait until completion

# COMMAND ----------

refreshes = w.quality_monitors.list_refreshes(table_name=f"{catalog}.{db}.hls_readmission_batch_inference").refreshes
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.get_refresh(table_name=f"{catalog}.{db}.hls_readmission_batch_inference", refresh_id=run_info.refresh_id)
  time.sleep(30)

assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"

# COMMAND ----------

w.quality_monitors.get(table_name=f"{catalog}.{db}.hls_readmission_batch_inference")

# COMMAND ----------

# DBTITLE 1,Delete existing monitor [OPTIONAL]
# w.quality_monitors.delete(table_name=f"{catalog}.{db}.advanced_churn_offline_inference", purge_artifacts=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect dashboard
# MAGIC
# MAGIC You can now inspect the monitoring dashboard that is automatically generated for you. Navigate to `[Catalog].[Schema].hls_readmission_inference_processed` in the __Catalog Explorer__, go to the __Quality__ tab and click on the __View dashboard__ button. Alternatively, you can check the dashboard saved in the workspace here: [hls_readmission_inference_processed Monitoring]($./monitoring/mmt_demos.hls_readmission_dbdemoinit.hls_readmission_inference_processed/)
# MAGIC
# MAGIC __Below are older version dashboard snapshots. You can skip and reference the link above.__
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/07_view_dashboard_button.png?raw=true" width="480">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC You can see the number of inferences being done before the first monitor refresh (the first refresh "window"), as well as the model performance metrics.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/07_model_inferences.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Scrolling further down to the section on __Prediction drift__, you can see the confusion matrix and the percentage of the model's predictions.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/07_confusion_matrix.png?raw=true" width="1200">
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC We do not observe any drift yet, as we only have the first refresh "window". We will simulate some drifted data in the next step and refresh the monitor against the newly captured data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next: Test for drift and trigger a model retrain
# MAGIC
# MAGIC Now, let explore how to detect drift on the inference data and define violations rules for triggering a model (re)train workflow.
# MAGIC
# MAGIC Next steps:
# MAGIC * [Detect drift and trigger model retrain]($./04.4c-Drift-Detection-Guanyu-Chen)
