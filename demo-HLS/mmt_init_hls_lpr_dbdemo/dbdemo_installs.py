# Databricks notebook source
## USE MLdbr instead!!!

%pip install dbdemos

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import dbdemos

# COMMAND ----------

dbdemos.list_demos()

# COMMAND ----------

# dbdemos.install('lakehouse-hls-readmission', catalog='mmt_demos', schema='hls_readmission_dbdemoinit')

# Installing demo lakehouse-hls-readmission under /Users/may.merkletan@databricks.com/REPOs/dbdemos-notebooks/demo-HLS/mmt_init_hls_lpr_dbdemo, please wait...
# Help us improving dbdemos, share your feedback or create an issue if something isn't working: https://github.com/databricks-demos/dbdemos


# COMMAND ----------



# COMMAND ----------

dbdemos.create_cluster('lakehouse-hls-readmission')

# COMMAND ----------

# DBTITLE 1,cluster config (create)
# {
#     "cluster_name": "dbdemos-lakehouse-hls-readmission-may_merkletan",
#     "spark_version": "15.3.x-cpu-ml-scala2.12",
#     "spark_conf": {
#         "spark.databricks.dataLineage.enabled": "true"
#     },
#     "aws_attributes": {
#         "first_on_demand": 1,
#         "availability": "SPOT_WITH_FALLBACK",
#         "spot_bid_price_percent": 100,
#         "ebs_volume_count": 0
#     },
#     "node_type_id": "i3.xlarge",
#     "custom_tags": {
#         "project": "dbdemos",
#         "demo": "lakehouse-hls-readmission"
#     },
#     "autotermination_minutes": 60,
#     "single_user_name": "may.merkletan@databricks.com",
#     "data_security_mode": "SINGLE_USER",
#     "runtime_engine": "STANDARD",
#     "autoscale": {
#         "min_workers": 2,
#         "max_workers": 10
#     }
# }
