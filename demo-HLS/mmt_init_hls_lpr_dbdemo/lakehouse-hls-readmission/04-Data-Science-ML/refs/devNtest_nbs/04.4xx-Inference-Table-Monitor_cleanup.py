# Databricks notebook source
# MAGIC %md
# MAGIC ### Clean up / `Delete` `old/existing` Monitor + Inference Tables/Resources before re-establishing new 

# COMMAND ----------

# %pip install "databricks-sdk>=0.28.0"

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,deleting existing Monitor | re-start
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

# w = WorkspaceClient()

# catalog, db, model_name = "mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr".split('.')
# # old serving_endpoint_name
# serving_endpoint_name = "dbdemos_hls_pr_endpoint"

# ## remove corresponding old inference_processed
# w.quality_monitors.delete(table_name=f"{catalog}.{db}.inference_processed")

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,clean up inference_[tables]
catalog, db, model_name = "mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr".split('.')

tables = spark.sql(f"SHOW TABLES IN {catalog}.{db}")
# display(tables)

# COMMAND ----------

# DBTITLE 1,find inference_* tables
display(tables.filter(F.col('tableName').rlike('inference')))

# COMMAND ----------

# DBTITLE 1,purge tables before
# tables2purge = [ 
#                 'inference_groundtruth', 
#                 'inference_payload',
#                 # 'inference_processed',
#                 # 'inference_processed_drift_metrics',
#                 # 'inference_processed_profile_metrics',
#                 # 'inference_unpacked'
#                ]

# tables2purge = [ 
#                 # 'inference_v2_groundtruth', 
#                 # 'inference_v2_payload',
#                 'inference_v2_processed',
#                 # 'inference_v2_processed_drift_metrics',
#                 # 'inference_v2_processed_profile_metrics',
#                 'inference_v2_unpacked'
#                ]

for table in tables2purge:
  print('table to drop: ', table)
  spark.sql(f"DROP TABLE IF EXISTS {catalog}.{db}.{table}")

# COMMAND ----------

# catalog, db, model_name = "mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr".split('.')

tables = spark.sql(f"SHOW TABLES IN {catalog}.{db}")
# display(tables)

display(tables.filter(F.col('tableName').rlike('inference')))

# COMMAND ----------


