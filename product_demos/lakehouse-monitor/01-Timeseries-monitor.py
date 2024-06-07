# Databricks notebook source
# MAGIC %md
# MAGIC # Lakehouse Monitoring Demo
# MAGIC
# MAGIC ### Use case
# MAGIC Let's explore a retail use case where one of the most important layers is the `silver_transaction` table that joins data from upstream bronze tables and impacts downstream gold tables. The data schema used in the demo is as follows: 
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/lhm/lhm_data.png" width="600px" style="float:right"/>
# MAGIC
# MAGIC Data analysts are using these tables to generate reports and make various business decisions. Most recently, an analyst is trying to determine the most popular `PreferredPaymentMethod`. When querying the `silver_transaction` table, they discover that there's been a number of transactions with `null` `PreferredPaymentMethod` and shares a screenshot of the problem:
# MAGIC
# MAGIC ![image3](https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/lhm/lhm_payment_type.png?raw=true)
# MAGIC
# MAGIC At this point, you may be asking yourself a number of questions such as: 
# MAGIC 1. What percent of `nulls` have been introduced to this column? Is this normal? 
# MAGIC 2. If it's not normal, what was the root cause for this integrity issue? 
# MAGIC 3. What are the downstream assets that might've been impacted by this issue?
# MAGIC
# MAGIC Let's explore how Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/en/lakehouse-monitoring/index.html)| [Azure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/)) can help you answer these types of questions. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.28.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/01-DataGeneration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. View the dataset
# MAGIC

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- To setup monitoring, load in the silver_transaction dataset
# MAGIC SELECT * from silver_transaction limit 100;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the monitor
# MAGIC
# MAGIC To create a monitor, we can choose from three different types of profile types: 
# MAGIC 1. **Timeseries**: Aggregates quality metrics over time windows
# MAGIC 2. **Snapshot**: Calculates quality metrics over the full table
# MAGIC 3. **Inference**: Tracks model drift and performance over time
# MAGIC
# MAGIC Since we are monitoring transaction data and have a timestamp column in the table, a Timeseries works best in this scenario. For other types of analysis, see the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/en/lakehouse-monitoring/create-monitor-ui.html#profiling)| [Azure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/create-monitor-ui#profiling)).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries
import os

# COMMAND ----------

# Define time windows to aggregate metrics over
GRANULARITIES = ["1 day"]                       

# Optionally define expressions to slice data with
# SLICING_EXPRS = ["Category='Toys'"]  

# COMMAND ----------

# You must have `USE CATALOG` privileges on the catalog, and you must have `USE SCHEMA` privileges on the schema.
# If necessary, change the catalog and schema name here.
TABLE_NAME = f"{catalog}.{dbName}.silver_transaction"

# Define the timestamp column name
TIMESTAMP_COL = "TransactionDate"

# Enable Change Data Feed (CDF) to incrementally process changes to the table and make execution more efficient
(spark.readStream.format("delta")
  .option("readChangeFeed", "true")
  .table(TABLE_NAME)
)

# COMMAND ----------

# Create a monitor using a Timeseries profile type. After the intial refresh completes, you can view the autogenerated dashboard from the Quality tab of the table in Catalog Explorer. 
print(f"Creating monitor for {TABLE_NAME}")

w = WorkspaceClient()

try:
  lhm_monitor = w.quality_monitors.create(
    table_name=TABLE_NAME, # Always use 3-level namespace
    time_series = MonitorTimeSeries(
      timestamp_col=TIMESTAMP_COL,
      granularities=GRANULARITIES
    ),
    assets_dir = os.getcwd(),
    output_schema_name=f"{catalog}.{dbName}"
  )

except Exception as lhm_exception:
  print(lhm_exception)

# COMMAND ----------

# Display profile metrics table
profile_table = f"{TABLE_NAME}_profile_metrics"
display(spark.sql(f"SELECT * FROM {profile_table}"))

# Display the drift metrics table
drift_table = f"{TABLE_NAME}_drift_metrics"
display(spark.sql(f"SELECT * FROM {drift_table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## View the Autogenerated Dashboard
# MAGIC After the intial refresh completes, you can view the autogenerated dashboard from the Quality tab of the `silver_transactions` table in Catalog Explorer. The dashboard visualizes metrics in the following sections: 
# MAGIC 1. **Data Volume**: Check if transaction volume is expected or if there's been changes with seasonality
# MAGIC 2. **Data Integrity**: Identify the columns with a high % of nulls or zeros and view their distribution over time
# MAGIC 3. **Numerical Distribution Change**: Identify numerical anomalies and view the Range of values over time
# MAGIC 4. **Categorical Distribution Change**: Identify categorical anomalies like `PreferredPaymentMethod` and view the distribution of values time
# MAGIC 5. **Profiling**: Explore the numerical and categorical data profile over time
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/lhm/lhm_dashboard-1.png?raw=true" width="800px" style="float:right"/>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Without any additional tools or complexity, Lakehouse Monitoring allows you to easily profile, diagnose, and enforce quality directly in the Databricks Data Intelligence Platform. 
# MAGIC
# MAGIC Based on the dashboard, we can answer the 3 questions that we originally had: 
# MAGIC
# MAGIC 1. What percent of nulls have been introduced to this column? Is this normal?
# MAGIC > From the % Nulls section, we can see that `PreferredPaymentMethod` spiked from 10% to around 40%. 
# MAGIC 2. If it's not normal, what was the root cause for this integrity issue?
# MAGIC > Using the Categorical Distribution Change section, we can see that both `PreferredPaymentMethod` and `PaymentMethod` had high drift in the last time window. With the heatmap, we can discover that Apple Pay was recently added as a new `PaymentMethod` at the same time `nulls` started to appear in `PreferredPaymentMethod`
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/lhm/lhm_dashboard-2.png?raw=true" width="800px" style="float:right"/>
# MAGIC 3. What are the downstream assets that might've been impacted by this issue?
# MAGIC > Since Lakehouse Monitoring is built on top of Unity Catalog, you can use the Lineage Graph to identify downstream tables that have been impacted: 
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/lhm/lhm_lineage.png?raw=true" width="800px" style="float:right"/>
# MAGIC
# MAGIC Like we explored in this demo, you can proactively discover quality issues before downstream processes are impacted. Get started with Lakehouse Monitoring (Generally Available) ([AWS](https://docs.databricks.com/en/lakehouse-monitoring/index.html)| [Azure](https://learn.microsoft.com/en-us/azure/databricks/lakehouse-monitoring/)) today and ensure reliability across your entire data + AI estate.