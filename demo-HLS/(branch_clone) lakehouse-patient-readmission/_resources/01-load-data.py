# Databricks notebook source
# dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC Load data for the demo

# COMMAND ----------

# DBTITLE 1,let's init the config
# MAGIC %run ../config

# COMMAND ----------

# DBTITLE 1,lets check the details
catalog, db ,volume_name

# COMMAND ----------

# DBTITLE 1,using updated_config
# MAGIC %run ./00-setup $reset_all_data=$reset_all_data $catalog=mmt_demos $db=hls_readmission_viarepo

# COMMAND ----------

# DBTITLE 1,original
# %run ./00-setup $reset_all_data=$reset_all_data $catalog=dbdemos $db=hls_patient_readmission

# COMMAND ----------


