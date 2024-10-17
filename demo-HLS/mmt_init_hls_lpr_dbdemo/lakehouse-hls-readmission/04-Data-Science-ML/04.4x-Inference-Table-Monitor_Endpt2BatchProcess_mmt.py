# Databricks notebook source
# MAGIC %md
# MAGIC ### Use `served model endpoint` for Batch Inference
# MAGIC
# MAGIC - generate pseudo sample_data for inference and corresponding ground-truth label(s) using `training_data`; save pseudo sample_data for subsequent `baseline` table joins 
# MAGIC
# MAGIC - make small batches of inferences at a certain frequency (e.g. 1hr)
# MAGIC
# MAGIC - [Scheduled NB Job](https://e2-demo-field-eng.cloud.databricks.com/jobs/4272949172505?o=1444828305810485)/code works with Serverless 

# COMMAND ----------

# DBTITLE 1,dependencies | libraries
# import pyspark.sql.functions as F
# from pyspark.sql.window import Window

from pyspark.sql import functions as F, types as T, window as w
import random

# COMMAND ----------

# DBTITLE 1,UC/endpoint variables
catalog, db, model_name = "mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr".split('.')
catalog, db, model_name
# serving_endpoint_name = "dbdemos_hls_pr_endpoint_v2" ## update

# COMMAND ----------

# DBTITLE 1,update schema/dtype for label
# df = spark.table(f'{catalog}.{db}.training_dataset')#.withColumn('30_DAY_READMISSION', F.col('30_DAY_READMISSION').cast(T.DoubleType()))
# # display(df)
# # spark.sql(f"drop table if exists {catalog}.{db}.training_dataset")
# df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{db}.training_dataset")

# COMMAND ----------

# DBTITLE 1,load training_dataset
#For this demo, reuse our dataset to test the batch inferences
training_sDF = spark.table(f'{catalog}.{db}.training_dataset')  

# COMMAND ----------

# DBTITLE 1,EDA on training_sDF
# training_sDF.select(F.min('START'), F.max('START'),
#                     F.min('STOP'), F.max('STOP'),
#                     ).show()
# +-------------------+-------------------+-------------------+-------------------+
# |         min(START)|         max(START)|          min(STOP)|          max(STOP)|
# +-------------------+-------------------+-------------------+-------------------+
# |1914-03-08 09:53:12|2023-06-30 03:52:44|1914-03-08 10:08:12|2023-06-30 04:07:44|
# +-------------------+-------------------+-------------------+-------------------+


# training_sDF.count()
# 142395

# training_sDF.groupby('ENCOUNTER_ID').agg(F.count('ENCOUNTER_ID').alias('encounter_count')).groupby('encounter_count').agg(F.count('*')).show()
# +---------------+--------+
# |encounter_count|count(1)|
# +---------------+--------+
# |              1|  142395|
# +---------------+--------+


# training_sDF.groupby('patient_id').agg(F.countDistinct('ENCOUNTER_ID')).show()
# patient with multiple encounters

# COMMAND ----------

# DBTITLE 1,featureCols
# dataset['dataframe_split']['columns']

featureCols = ['MARITAL_M',
               'MARITAL_S',
               'RACE_asian',
               'RACE_black',
               'RACE_hawaiian',
               'RACE_other',
               'RACE_white',
               'ETHNICITY_hispanic',
               'ETHNICITY_nonhispanic',
               'GENDER_F',
               'GENDER_M',
               'INCOME',
               'BASE_ENCOUNTER_COST',
               'TOTAL_CLAIM_COST',
               'PAYER_COVERAGE',
               'enc_length',
               'ENCOUNTERCLASS_ambulatory',
               'ENCOUNTERCLASS_emergency',
               'ENCOUNTERCLASS_hospice',
               'ENCOUNTERCLASS_inpatient',
               'ENCOUNTERCLASS_outpatient',
               'ENCOUNTERCLASS_wellness',
               'age_at_encounter']

# COMMAND ----------

# DBTITLE 1,inferenceCols
inferenceCols = ['patient_id', 'ENCOUNTER_ID', '30_DAY_READMISSION']

# featureCols + inferenceCols

# COMMAND ----------

# DBTITLE 1,generate random sampleInf_sDF
sampleInf_sDF = (training_sDF.sample(False, 0.2)  #seed= 9864
                 .select(*(featureCols + inferenceCols))
                 .withColumn("current_datetime", F.current_timestamp()) 
                 .withColumn("current_timestamp", F.to_unix_timestamp("current_datetime"))
                 .limit(random.sample(range(2,10), 1)[0]) # generate a random number between 2 and 10 and use it as the limit
                 .withColumn('30_DAY_READMISSION', F.col('30_DAY_READMISSION').cast(T.DoubleType())) ## same schema/dtype as predictions
                 )

display(sampleInf_sDF)

# COMMAND ----------

# sampleInf_sDF.printSchema()

# COMMAND ----------

# DBTITLE 1,test Inference input formatting
## (test_dataset.sample(False, 0.1).limit(5)).toPandas().to_dict(orient='split')
## sampleInf_sDF.toPandas().to_dict(orient='split')

# import json
# ds_dict = sampleInf_sDF.select(*featureCols+inferenceCols[:-1]).toPandas().to_dict(orient='split')
# data_json = json.dumps(ds_dict, allow_nan=True)
# ds_dict, data_json

# COMMAND ----------

# DBTITLE 1,Define Model Scoring Functions
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}
        
def score_model(dataset, access_token):
    # url = f'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/{serving_endpoint_name}/invocations'
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/dbdemos_hls_pr_endpoint_v2/invocations'

    # Include the access token in the request headers
    headers = {"Authorization": f"Bearer {access_token}","Content-Type": "application/json"}
    
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    
    data_json = json.dumps(ds_dict, allow_nan=True)

    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


# Wrapper function to create a pandas UDF that includes the access_token
def get_batch_predict_udf(access_token):
    @F.pandas_udf(T.DoubleType()) ## schema/dtype of predictions output
    def batch_predict(batch_pd: pd.DataFrame) -> pd.Series:
        # Your model scoring logic here, using access_token as needed
        predictions = score_model(batch_pd, access_token)
        return pd.Series(predictions['predictions'])
    return batch_predict

access_token = dbutils.secrets.get(scope='mmt', key='DATABRICKS_TOKEN')  # Define your access token
batch_predict_udf = get_batch_predict_udf(access_token)


# COMMAND ----------

df = sampleInf_sDF
display(df)

# COMMAND ----------

# DBTITLE 1,Apply Batch processing with served endpoint

# Apply the UDF to the spark DataFrame 
# feature_cols = [*featureCols+inferenceCols[:-1]]  # Replace with your actual feature column names
feature_cols = [*featureCols, *inferenceCols] ## keep the 30_DAY_READMISSION

# Use F.col to refer to DataFrame columns
df = df.withColumn("prediction", batch_predict_udf(F.struct(*feature_cols)))

display(df)

# COMMAND ----------

# DBTITLE 1,Save as inference_[v#]_groundtruth

df.write.format("delta").mode("append").saveAsTable(f"{catalog}.{db}.inference_v2_batchpredictions", mergeSchema=True)

# COMMAND ----------

# spark.sql(f""" Drop table if exists {catalog}.{db}.inference_baseline; """)
# spark.sql(f""" Drop table if exists {catalog}.{db}.inference_v2_baseline; """)
# spark.sql(f"drop table if exists {catalog}.{db}.inference_v2_groundtruth")

# COMMAND ----------

# DBTITLE 1,update dtypes for columns
# catalog, db, model_name = "mmt_demos.hls_readmission_dbdemoinit.dbdemos_hls_pr".split('.')
# bDF = spark.table(f"{catalog}.{db}.inference_v2_groundtruth")#.withColumn('30_DAY_READMISSION', F.col('30_DAY_READMISSION').cast(T.DoubleType()))

# display(bDF)
# bDF.count()

# spark.sql(f"drop table if exists {catalog}.{db}.inference_v2_groundtruth")
# (bDF
#  .write.format("delta").option("mergeSchema", "true")
#  .mode("overwrite")
#  .saveAsTable(f"{catalog}.{db}.inference_v2_batchpredictions")
# )


# COMMAND ----------


