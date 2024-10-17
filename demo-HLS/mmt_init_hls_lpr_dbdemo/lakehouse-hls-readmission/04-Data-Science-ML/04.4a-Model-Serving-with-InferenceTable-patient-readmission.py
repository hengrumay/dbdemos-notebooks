# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Getting realtime patient risks
# MAGIC
# MAGIC Let's leverage the model we trained to deploy real-time inferences behind a REST API.
# MAGIC
# MAGIC This will provide instant recommandations for any new patient, on demand, potentially also explaining the recommendation (see [next notebook]($./03.5-Explainability-patient-readmission) for Explainability) 
# MAGIC
# MAGIC Now that our model has been created with Databricks AutoML, we can easily flag it as Production Ready and turn on Databricks Model Serving.
# MAGIC
# MAGIC We'll be able to send HTTP REST Requests and get inference (risk probability) in real-time.
# MAGIC
# MAGIC
# MAGIC ## Databricks Model Serving
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/hls/patient-readmission/patient-risk-ds-flow-4.png?raw=true" width="700px" style="float: right; margin-left: 10px;" />
# MAGIC
# MAGIC
# MAGIC Databricks Model Serving is fully serverless:
# MAGIC
# MAGIC * One-click deployment. Databricks will handle scalability, providing blazing fast inferences and startup time.
# MAGIC * Scale down to zero as an option for best TCO (will shut down if the endpoint isn't used).
# MAGIC * Built-in support for multiple models & version deployed.
# MAGIC * A/B Testing and easy upgrade, routing traffic between each versions while measuring impact.
# MAGIC * Built-in metrics & monitoring.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=lakehouse&org_id=1444828305810485&notebook=%2F04-Data-Science-ML%2F04.4-Model-Serving-patient-readmission&demo_name=lakehouse-hls-readmission&event=VIEW&path=%2F_dbdemos%2Flakehouse%2Flakehouse-hls-readmission%2F04-Data-Science-ML%2F04.4-Model-Serving-patient-readmission&version=1">

# COMMAND ----------

# DBTITLE 1,Make sure we have the latset sdk (used in the helper)
# MAGIC %pip install databricks-sdk -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Load the model with "prod" alias from Unity Catalog Registry
# model_name = "dbdemos_hls_patient_readmission"
model_name = "dbdemos_hls_pr"

full_model_name = f"{catalog}.{db}.{model_name}"

import mlflow
from mlflow import MlflowClient

#Enable Unity Catalog with mlflow registry
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

#Get model with PROD alias (make sure you run the notebook 04.2 to save the model in UC)
latest_model = client.get_model_version_by_alias(full_model_name, "prod")
print(latest_model)

# COMMAND ----------

# https://docs.databricks.com/en/machine-learning/model-serving/enable-model-serving-inference-tables.html

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

# serving_endpoint_name = "dbdemos_hls_patient_readmission_endpoint"
# serving_endpoint_name = "dbdemos_hls_pr_endpoint"
serving_endpoint_name = "dbdemos_hls_pr_endpoint_v2"

w = WorkspaceClient()

endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=full_model_name,
            entity_version=latest_model.version,
            scale_to_zero_enabled=True,
            workload_size="Small"
        )
    ], ## Added to enable auto-capture / inference table -- mmt 2024Oct03
    auto_capture_config=AutoCaptureConfigInput(
       catalog_name=catalog, # from config/setup
       schema_name=db, # from config/setup
    #    table_name_prefix="inference",
       table_name_prefix="inference_v2",
       enabled=True  # This is optional and true by default
    )
)

#Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
force_update = False 

try:
  existing_endpoint = w.serving_endpoints.get(serving_endpoint_name)
  print(f"endpoint {serving_endpoint_name} already exist - force update = {force_update}...")

  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)
    
except:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)


if existing_endpoint.state == 'READY':
    # Construct the endpoint URL
    workspace_url = "https://<your-workspace-url>"
    endpoint_id = existing_endpoint.id
    endpoint_url = f"{workspace_url}/model/{serving_endpoint_name}/{endpoint_id}/serve"
    print(f"The endpoint URL is: {endpoint_url}")
else:
    print(f"The endpoint {serving_endpoint_name} is not ready. Current state: {existing_endpoint.state}")

# COMMAND ----------

# import time

# max_retries = 3  # Maximum number of retries
# retry_delay = 600  # Delay between retries in seconds (10 minutes)

# for attempt in range(max_retries):
#     try:
#         existing_endpoint = w.serving_endpoints.get(serving_endpoint_name)
#         print(f"Endpoint {serving_endpoint_name} already exists - force update = {force_update}...")
#         if force_update:
#             w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)
#         break  # If the operation is successful, exit the loop
#     except TimeoutError as e:
#         print(f"Attempt {attempt + 1} timed out. Retrying in {retry_delay} seconds...")
#         time.sleep(retry_delay)
#     except Exception as e:
#         print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
#         w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
#         break  # Assuming creation is a one-time attempt if it doesn't exist

# COMMAND ----------

# MAGIC %md 
# MAGIC Our model endpoint was automatically created. 
# MAGIC
# MAGIC <!-- Open the [endpoint UI](#mlflow/endpoints/dbdemos_hls_pr_endpoint) to explore your endpoint and use the UI to send queries.    -->
# MAGIC Open the [endpoint UI](#mlflow/endpoints/dbdemos_hls_pr_endpoint_v2) to explore your endpoint and use the UI to send queries. 
# MAGIC **!!! Refer to cell below for json input example**
# MAGIC
# MAGIC *Note that the first deployment will build your model image and take a few minutes. It'll then stop & start instantly.*

# COMMAND ----------

# DBTITLE 1,example patient_id | ENCOUNTER_ID
# b92cc118-75a3-e3d5-b00c-1c272cfdbb9d	69e72f2b-adbe-a638-f561-790bebc5b78a
# 73801a01-8b51-3d37-8667-c77d588fbfb1	10e43172-a220-b7b6-be39-3ca31e7d092f
# 332bcba3-dd9a-2e75-4445-52b16c934990	a4668589-5622-88ff-26fc-0035b0e63a87
# 626f0810-f1d4-5462-15b4-a82c76f6b398	5ab73642-6eb5-1297-7cdd-ce0015ee4992
# dded39b5-0dc8-1be7-0bdd-eafd4b9ffa37	1cd53dc3-a0da-2d61-3cb6-de62f75dd050

# COMMAND ----------

# DBTITLE 1,Adding extra columns to the inference
# "patient_id", ## extra non-feature input for joining with baseline
# "ENCOUNTER_ID", ## extra non-feature input for joining with baseline

# COMMAND ----------

# DBTITLE 1,USE example json: For Querying Endpoint
## example data_json *

#  {"dataframe_split": {"index": [0, 1, 2, 3, 4, 5, 6, 7, 8], "columns": ["MARITAL_M", "MARITAL_S", "RACE_asian", "RACE_black", "RACE_hawaiian", "RACE_other", "RACE_white", "ETHNICITY_hispanic", "ETHNICITY_nonhispanic", "GENDER_F", "GENDER_M", "INCOME", "BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE", "enc_length", "ENCOUNTERCLASS_ambulatory", "ENCOUNTERCLASS_emergency", "ENCOUNTERCLASS_hospice", "ENCOUNTERCLASS_inpatient", "ENCOUNTERCLASS_outpatient", "ENCOUNTERCLASS_wellness", "age_at_encounter", "patient_id", "ENCOUNTER_ID"], "data": [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 892590, 85.55, 755.56, 755.56, 13500, 1, 0, 0, 0, 0, 0, 45.65366187542779, "e4ff91bc-bd28-3820-8e03-7966fdd0289f", "6730a65f-470c-a2c5-46bf-6c31bd3ab0dc"], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 55722, 136.8, 777.55, 0.0, 900, 0, 0, 0, 0, 0, 1, 1.4373716632443532, "9f1c79c3-eb93-96db-292f-990b4084fda8", "c4827fef-7583-973f-5f6b-09f1b66555f2"], [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 54368, 136.8, 979.11, 0.0, 900, 0, 0, 0, 0, 0, 1, 0.6899383983572895, "d44b76b7-88de-d9a0-d6dd-2395b8552739", "f3742f2c-da43-ceb0-2201-f561551033a3"], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 36123, 136.8, 791.94, 0.0, 2771, 0, 0, 0, 0, 0, 1, 50.116358658453116, "293ffee1-e1a5-1511-746e-2f55108d7985", "1d5728f3-b794-2307-3f2a-442f8e3e7442"], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 99024, 136.8, 1021.83, 817.46, 900, 0, 0, 0, 0, 0, 1, 3.9288158795345653, "35aaa13d-b480-a1b7-770e-0d02c8990341", "44a19292-e7ff-abed-f82b-b9211e0e1eae"], [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 169036, 142.58, 8646.42, 3149.49, 3004, 0, 0, 0, 0, 1, 0, 28.15605749486653, "ce3af08b-821d-5288-0e5a-88950f1265d1", "41d4cdc2-b215-e210-7066-bbc5d37ec648"], [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 28150, 136.8, 923.8, 0.0, 2378, 0, 0, 0, 0, 0, 1, 37.21834360027378, "12e088db-4ad6-5b3c-ae25-1b3103fd6585", "1a35fce9-6f4c-a40c-5daa-6209c90d5ed5"], [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 149217, 142.58, 11031.76, 11031.76, 900, 1, 0, 0, 0, 0, 0, 32.733744010951405, "83e4b8b0-1f23-fb55-294c-3b74159cd40e", "518a20c5-60fa-6732-8b24-342acc8bc77b"], [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 123273, 136.8, 9833.74, 7866.99, 3178, 0, 0, 0, 0, 0, 1, 25.182751540041068, "8e88c649-1f18-d322-9fd8-55e35cd014de", "57c8ef93-315c-3829-3016-35810c33df20"]]}}
 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Testing the model
# MAGIC
# MAGIC Now that the model is deployed, let's test it with information from one of our patient. *Note that we could also chose to return a risk percentage instead of a binary result.*

# COMMAND ----------

# p = ModelsArtifactRepository(f"models:/{full_model_name}@prod").download_artifacts("") 
# dataset =  {"dataframe_split": Model.load(p).load_input_example(p).to_dict(orient='split')}
# dataset

# COMMAND ----------

dataset = {"dataframe_split": {"index": [0, 1, 2, 3, 4, 5, 6, 7, 8], "columns": ["MARITAL_M", "MARITAL_S", "RACE_asian", "RACE_black", "RACE_hawaiian", "RACE_other", "RACE_white", "ETHNICITY_hispanic", "ETHNICITY_nonhispanic", "GENDER_F", "GENDER_M", "INCOME", "BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE", "enc_length", "ENCOUNTERCLASS_ambulatory", "ENCOUNTERCLASS_emergency", "ENCOUNTERCLASS_hospice", "ENCOUNTERCLASS_inpatient", "ENCOUNTERCLASS_outpatient", "ENCOUNTERCLASS_wellness", "age_at_encounter", "patient_id", "ENCOUNTER_ID"], "data": [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 892590, 85.55, 755.56, 755.56, 13500, 1, 0, 0, 0, 0, 0, 45.65366187542779, "e4ff91bc-bd28-3820-8e03-7966fdd0289f", "6730a65f-470c-a2c5-46bf-6c31bd3ab0dc"], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 55722, 136.8, 777.55, 0.0, 900, 0, 0, 0, 0, 0, 1, 1.4373716632443532, "9f1c79c3-eb93-96db-292f-990b4084fda8", "c4827fef-7583-973f-5f6b-09f1b66555f2"], [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 54368, 136.8, 979.11, 0.0, 900, 0, 0, 0, 0, 0, 1, 0.6899383983572895, "d44b76b7-88de-d9a0-d6dd-2395b8552739", "f3742f2c-da43-ceb0-2201-f561551033a3"], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 36123, 136.8, 791.94, 0.0, 2771, 0, 0, 0, 0, 0, 1, 50.116358658453116, "293ffee1-e1a5-1511-746e-2f55108d7985", "1d5728f3-b794-2307-3f2a-442f8e3e7442"], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 99024, 136.8, 1021.83, 817.46, 900, 0, 0, 0, 0, 0, 1, 3.9288158795345653, "35aaa13d-b480-a1b7-770e-0d02c8990341", "44a19292-e7ff-abed-f82b-b9211e0e1eae"], [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 169036, 142.58, 8646.42, 3149.49, 3004, 0, 0, 0, 0, 1, 0, 28.15605749486653, "ce3af08b-821d-5288-0e5a-88950f1265d1", "41d4cdc2-b215-e210-7066-bbc5d37ec648"], [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 28150, 136.8, 923.8, 0.0, 2378, 0, 0, 0, 0, 0, 1, 37.21834360027378, "12e088db-4ad6-5b3c-ae25-1b3103fd6585", "1a35fce9-6f4c-a40c-5daa-6209c90d5ed5"], [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 149217, 142.58, 11031.76, 11031.76, 900, 1, 0, 0, 0, 0, 0, 32.733744010951405, "83e4b8b0-1f23-fb55-294c-3b74159cd40e", "518a20c5-60fa-6732-8b24-342acc8bc77b"], [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 123273, 136.8, 9833.74, 7866.99, 3178, 0, 0, 0, 0, 0, 1, 25.182751540041068, "8e88c649-1f18-d322-9fd8-55e35cd014de", "57c8ef93-315c-3829-3016-35810c33df20"]]}}

# COMMAND ----------

serving_endpoint_name

# COMMAND ----------

import mlflow
from mlflow import deployments
deployment_client = mlflow.deployments.get_deploy_client("databricks")
predictions = deployment_client.predict(endpoint=serving_endpoint_name, inputs=dataset)

print(f"Patient readmission risk: {predictions}.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Updating your model and monitoring its performance with A/B testing 
# MAGIC
# MAGIC Databricks Model Serving let you easily deploy & test new versions of your model.
# MAGIC
# MAGIC You can dynamically reconfigure your endpoint to route a subset of your traffic to a newer version. In addition, you can leverage endpoint monitoring to understand your model behavior and track your A/B deployment.
# MAGIC
# MAGIC * Without making any production outage
# MAGIC * Slowly routing requests to the new model
# MAGIC * Supporting auto-scaling & potential bursts
# MAGIC * Performing some A/B testing ensuring the new model is providing better outcomes
# MAGIC * Monitorig our model outcome and technical metrics (CPU/load etc)
# MAGIC
# MAGIC Databricks makes this process super simple with Serverless Model Serving endpoint.
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Model monitoring and A/B testing analysis
# MAGIC
# MAGIC Because the Model Serving runs within our Lakehouse, Databricks will automatically save and track all our Model Endpoint results as a Delta Table.
# MAGIC
# MAGIC We can then easily plug a feedback loop to start analysing the revenue in $ each model is offering. 
# MAGIC
# MAGIC All these metrics, including A/B testing validation (p-values etc) can then be pluged into a Model Monitoring Dashboard and alerts can be sent for errors, potentially triggering new model retraining or programatically updating the Endpoint routes to fallback to another model.
# MAGIC
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/model-serving-monitoring.png" width="1200px" />

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Next
# MAGIC
# MAGIC Making sure your model doesn't have bias and being able to explain its behavior is extremely important to increase health care quality and personalize patient journey. <br/>
# MAGIC Explore your model with [04.5-Explainability-patient-readmission]($./04.5-Explainability-patient-readmission) on the Lakehouse.
# MAGIC
# MAGIC ## Conclusion: the power of the Lakehouse
# MAGIC
# MAGIC In this demo, we've seen an end 2 end flow with the Lakehouse:
# MAGIC
# MAGIC - Data ingestion made simple with Delta Live Table
# MAGIC - Leveraging Databricks notebooks and SQL warehouse to create, anaylize and share our dashboards 
# MAGIC - Model Training with AutoML for citizen Data Scientist
# MAGIC - Ability to tune our model for better results, improving our patient journey quality
# MAGIC - Ultimately, the ability to deploy and make explainable ML predictions, made possible with the full Lakehouse capabilities.
# MAGIC
# MAGIC [Go back to the introduction]($../00-patient-readmission-introduction) or discover how to use Databricks Workflow to orchestrate everything together through the [05-Workflow-Orchestration-patient-readmission]($../05-Workflow-Orchestration/05-Workflow-Orchestration-patient-readmission).
