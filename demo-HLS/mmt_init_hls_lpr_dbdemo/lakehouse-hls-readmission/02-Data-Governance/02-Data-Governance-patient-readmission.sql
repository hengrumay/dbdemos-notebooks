-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC # Unify Governance and security for all users and all data
-- MAGIC
-- MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/hls/patient-readmission/hls-patient-readmision-flow-2.png" style="float: right; margin-left: 10px; margin-top:10px" width="650px" />
-- MAGIC
-- MAGIC Our dataset contains sensitive information on our patients and admission.
-- MAGIC
-- MAGIC It's critical to be able to add a security and privacy layer on top of our data, but also on all the other data assets that will use this data (notebooks dashboards, Models, files etc).
-- MAGIC
-- MAGIC This is made easy with Databricks Unity Catalog.
-- MAGIC
-- MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
-- MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=lakehouse&org_id=1444828305810485&notebook=%2F02-Data-Governance%2F02-Data-Governance-patient-readmission&demo_name=lakehouse-hls-readmission&event=VIEW&path=%2F_dbdemos%2Flakehouse%2Flakehouse-hls-readmission%2F02-Data-Governance%2F02-Data-Governance-patient-readmission&version=1">

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Implementing a global data governance and security with Unity Catalog
-- MAGIC
-- MAGIC Let's see how the Lakehouse can solve this challenge leveraging Unity Catalog.
-- MAGIC
-- MAGIC Our Data has been saved as Delta Table by our Data Engineering team.  The next step is to secure this data while allowing cross team to access it. <br>
-- MAGIC A typical setup would be the following:
-- MAGIC
-- MAGIC * Data Engineers / Jobs can read and update the main data/schemas (ETL part)
-- MAGIC * Data Scientists can read the final tables and update their features tables
-- MAGIC * Data Analyst have READ access to the Data Engineering and Feature Tables and can ingest/transform additional data in a separate schema.
-- MAGIC * Data is masked/anonymized dynamically based on each user access level
-- MAGIC
-- MAGIC This is made possible by Unity Catalog. When tables are saved in the Unity Catalog, they can be made accessible to the entire organization, cross-workpsaces and cross users.
-- MAGIC
-- MAGIC Unity Catalog is key for data governance, including creating data products or organazing teams around datamesh. It brings among other:
-- MAGIC
-- MAGIC * Fined grained ACL, Row and Column Masking 
-- MAGIC * Audit log,
-- MAGIC * Data lineage,
-- MAGIC * Data exploration & discovery,
-- MAGIC * Sharing data with external organization (Delta Sharing),
-- MAGIC * (*coming soon*) Attribute-based access control. 
-- MAGIC
-- MAGIC ref: https://docs.databricks.com/en/tables/row-and-column-filters.html#

-- COMMAND ----------

-- MAGIC %run ../_resources/00-setup $reset_all_data=false

-- COMMAND ----------

SELECT CURRENT_CATALOG(), CURRENT_DATABASE();

-- COMMAND ----------

-- DBTITLE 1,As you can see, our tables are available under our catalog.
SHOW TABLES;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Step 1. Access control
-- MAGIC
-- MAGIC In the Lakehouse, you can use simple SQL GRANT and REVOKE statements to create granular (on data and even schema and catalog levels) access control irrespective of the data source or format.

-- COMMAND ----------

-- DBTITLE 1,original
-- Let's grant our ANALYSTS a SELECT permission:
-- Note: make sure you created an analysts and dataengineers group first.
GRANT SELECT ON TABLE drug_exposure TO `analysts`;
GRANT SELECT ON TABLE condition_occurrence TO `analysts`;
GRANT SELECT ON TABLE patients TO `analysts`;

-- We'll grant an extra MODIFY to our Data Engineer
-- GRANT SELECT, MODIFY ON SCHEMA dbdemos_hls_readmission TO `dataengineers`;
GRANT SELECT, MODIFY ON SCHEMA hls_readmission_dbdemoinit TO `dataengineers`;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Step 2. PII data masking, row and column-level filtering
-- MAGIC
-- MAGIC In the cells below we will demonstrate how to handle sensitive data through column and row masking.

-- COMMAND ----------

-- DBTITLE 1,show table `patients`
SELECT * FROM patients;

-- COMMAND ----------

-- DBTITLE 0,notes
-- MAGIC %md
-- MAGIC ##### Create a simple_mask function using SQL to help
-- MAGIC <!-- ## From previous version -- azure link https://adb-830292400663869.9.azuredatabricks.net/?o=830292400663869#notebook/2919862720218566/command/2919862720218586 -->

-- COMMAND ----------

-- DBTITLE 1,orignal code [doesn't work with streaming tables]
-- hls_admin group will have access to all data, all other users will see a masked information.
CREATE OR REPLACE FUNCTION simple_mask(column_value STRING)
   RETURN IF(is_account_group_member('hls_admin'), column_value, "****");
   
-- ALTER FUNCTION simple_mask OWNER TO `account users`; -- grant access to all user to the function for the demo - don't do it in production

-- Mask all PII information
ALTER TABLE patients ALTER COLUMN FIRST SET MASK simple_mask;
ALTER TABLE patients ALTER COLUMN LAST SET MASK simple_mask;
ALTER TABLE patients ALTER COLUMN PASSPORT SET MASK simple_mask;
ALTER TABLE patients ALTER COLUMN DRIVERS SET MASK simple_mask;
ALTER TABLE patients ALTER COLUMN SSN SET MASK simple_mask;
ALTER TABLE patients ALTER COLUMN ADDRESS SET MASK simple_mask;

SELECT * FROM patients

-- COMMAND ----------

-- DBTITLE 1,SQL functions
-- Create a function to mask data
CREATE OR REPLACE FUNCTION simple_mask(column_value STRING)
RETURNS STRING
RETURN IF(is_account_group_member('hls_admin'), column_value, '****');

-- Create a view to mask PII information
CREATE OR REPLACE VIEW masked_patients AS
SELECT
  simple_mask(FIRST) AS FIRST,
  simple_mask(LAST) AS LAST,
  simple_mask(PASSPORT) AS PASSPORT,
  simple_mask(DRIVERS) AS DRIVERS,
  simple_mask(SSN) AS SSN,
  simple_mask(ADDRESS) AS ADDRESS,
  -- Include other columns as needed
  -- other_column1,
  -- other_column2,
  BIRTHDATE,
  BIRTHPLACE,
  CITY,
  COUNTY
FROM patients;

-- Query the view
SELECT * FROM masked_patients;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC As we can observe from the cells above, the ```first_name``` column is masked whenever the current user requesting the data is part of the ```data-science-users``` group, and not masked if other type of users queries the data.

-- COMMAND ----------

-- DBTITLE 1,using pyspark.sql
-- MAGIC %python
-- MAGIC from pyspark.sql.functions import expr
-- MAGIC
-- MAGIC df = spark.table('patients')
-- MAGIC display(df)
-- MAGIC
-- MAGIC # Assuming `df` is your streaming DataFrame
-- MAGIC masked_df = df.withColumn("FIRST", expr("IF(is_account_group_member('hls_admin'), FIRST, '****')")) \
-- MAGIC               .withColumn("LAST", expr("IF(is_account_group_member('hls_admin'), LAST, '****')")) \
-- MAGIC               .withColumn("PASSPORT", expr("IF(is_account_group_member('hls_admin'), PASSPORT, '****')")) \
-- MAGIC               .withColumn("DRIVERS", expr("IF(is_account_group_member('hls_admin'), DRIVERS, '****')")) \
-- MAGIC               .withColumn("SSN", expr("IF(is_account_group_member('hls_admin'), SSN, '****')")) \
-- MAGIC               .withColumn("ADDRESS", expr("IF(is_account_group_member('hls_admin'), ADDRESS, '****')"))
-- MAGIC
-- MAGIC display(masked_df)

-- COMMAND ----------

-- DBTITLE 0,using pyspark.sql
-- %python
-- # from pyspark.sql.functions import col, lit, when
-- from pyspark.sql import functions as F, types as T


-- # Modified function to mask column without checking `is_hls_admin`
-- def mask_column(column_name):
--     return F.lit("****").alias(column_name)

-- # # Register the UDF
-- # mask_column_udf = F.udf(mask_column, T.StringType())

-- df = spark.table('patients')
-- display(df)


-- # Apply the modified function
-- masked_df = df.withColumn("FIRST", mask_column("FIRST")) \
--               .withColumn("LAST", mask_column("LAST")) \
--               .withColumn("PASSPORT", mask_column("PASSPORT")) \
--               .withColumn("DRIVERS", mask_column("DRIVERS")) \
--               .withColumn("SSN", mask_column("SSN")) \
--               .withColumn("ADDRESS", mask_column("ADDRESS"))

-- display(masked_df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Step 3. (Data and assets) Lineage
-- MAGIC
-- MAGIC Lineage is critical for understanding compliance, audit, observability, but also discoverability of data.
-- MAGIC
-- MAGIC These are three very common schenarios, where full data lineage becomes incredibly important:
-- MAGIC 1. **Explainability** - we need to have the means of tracing features used in machine learning to the raw data that created those features,
-- MAGIC 2. Tracing **missing values** in a dashboard or ML model to the origin,
-- MAGIC 3. **Finding specific data** - organizations have hundreds and even thousands of data tables and sources. Finiding the table or column that contains specific information can be daunting without a proper discoverability tools.
-- MAGIC
-- MAGIC In the image below, you can see every possible data (both ingested and created internally) in the same lineage graph, irrespective of the data type (stream vs batch), file type (csv, json, xml), language (SQL, python), or tool used (DLT, SQL query, Databricks Feature Store, or a python Notebook).
-- MAGIC
-- MAGIC **Note**: To explore the whole lineage, open navigate to the Data Explorer, and find the ```condition_occurrence``` | ```drug_exposure``` | *`gold`* *medallion* table inside your catalog and database.
-- MAGIC
-- MAGIC
-- MAGIC <!-- **Note**: To explore the whole lineage, open navigate to the Data Explorer, and find the ```customer_gold``` table inside your catalog and database. -->

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ### 4. Secure data sharing
-- MAGIC
-- MAGIC Once our data is ready, we can easily share it leveraging Delta Sharing, an open protocol to share your data assets with any customer or partnair.
-- MAGIC
-- MAGIC For more details on Delta Sharing, run `dbdemos.install('delta-sharing-airlines')`

-- COMMAND ----------

-- DBTITLE 0,Create a Delta Sharing Share (original)
-- CREATE SHARE IF NOT EXISTS dbdemos_patient_readmission_visits 
--   COMMENT 'Sharing patients table from the hls_readmissions Demo.';
 
-- -- For the demo we'll grant ownership to all users. Typical deployments wouls have admin groups or similar.
-- ALTER SHARE dbdemos_patient_readmission_visits OWNER TO `account users`;

-- -- Simply add the tables you want to share to your SHARE:
-- -- ALTER SHARE dbdemos_patient_readmission_visits  ADD TABLE patients ;

-- -- DESCRIBE SHARE dbdemos_patient_readmission_visits;

-- COMMAND ----------

-- DBTITLE 1,Create a Delta Sharing Share

-- Grant USE CATALOG privilege to the user
GRANT USE CATALOG ON CATALOG mmt_demos TO `account users`;

-- Grant USE SCHEMA privilege to the user
GRANT USE SCHEMA ON SCHEMA mmt_demos.hls_readmission_dbdemoinit TO `account users`;

-- Grant SELECT privilege on the table to the user
GRANT SELECT ON TABLE mmt_demos.hls_readmission_dbdemoinit.patients TO `account users`;



-- Create the share if it does not exist
CREATE SHARE IF NOT EXISTS mmt_dbdemos_hls_readmission_dbdemoinit_share 
  COMMENT 'Sharing patients table from the hls_readmission_dbdemoinit Demo.';

-- Grant ownership to all users
ALTER SHARE mmt_dbdemos_hls_readmission_dbdemoinit_share OWNER TO `account users`;

-- Add the table to the share
ALTER SHARE mmt_dbdemos_hls_readmission_dbdemoinit_share ADD TABLE mmt_demos.hls_readmission_dbdemoinit.patients;

-- COMMAND ----------

DESCRIBE SHARE mmt_dbdemos_hls_readmission_dbdemoinit_share;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Refs: 
-- MAGIC - [mmt_dbdemos_hls_readmission_dbdemoinit_share](https://e2-demo-field-eng.cloud.databricks.com/explore/sharing/shares/mmt_dbdemos_hls_readmission_dbdemoinit_share?o=1444828305810485)
-- MAGIC - https://docs.databricks.com/en/delta-sharing/create-recipient.html

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Next: Start Data Analysis on top of our existing dataset
-- MAGIC
-- MAGIC Our data is now ingested, secured, and our Data Scientist can access it.
-- MAGIC
-- MAGIC
-- MAGIC Let's get the maximum value out of the data we ingested: open the [Data Science and Analysis notebook]($../03-Data-Analysis-BI-Warehousing/03-Data-Analysis-BI-Warehousing-patient-readmission) and start building our patient cohorts.
-- MAGIC
-- MAGIC Go back to the [Introduction]($../00-patient-readmission-introduction).
