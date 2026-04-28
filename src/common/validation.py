import os
import sys
from google.cloud import storage
import great_expectations as gx
from pyspark.sql import DataFrame
from loguru import logger

# Correct path for GE context
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
BUCKET_NAME = "finance-data-lake-unique-id"

def sync_to_gcs():
    """
    Synchronizes local Data Docs to GCS using the Google Cloud Storage Python SDK.
    This ensures compatibility in environments without the gcloud CLI (like Docker).
    """
    gx_dir = os.path.join(ROOT_DIR, "gx_workspace", "gx")
    local_docs = os.path.join(gx_dir, "uncommitted", "data_docs", "local_site")
    
    if not os.path.exists(local_docs):
        logger.warning(f"⚠️ Local Data Docs not found at {local_docs}. Skipping sync.")
        return

    logger.info(f"📤 Syncing Data Docs to gs://{BUCKET_NAME}/data_quality_reports...")
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        prefix = "data_quality_reports"

        # Walk through local files and upload
        for root, dirs, files in os.walk(local_docs):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_docs)
                blob_path = f"{prefix}/{relative_path}"
                
                blob = bucket.blob(blob_path)
                
                # Set content type for HTML/CSS/JS
                content_type = None
                if filename.endswith(".html"): content_type = "text/html"
                elif filename.endswith(".css"): content_type = "text/css"
                elif filename.endswith(".js"): content_type = "application/javascript"
                
                blob.upload_from_filename(local_path, content_type=content_type)
        
        logger.success("✅ Data Docs synced to GCS via Python SDK.")
    except Exception as e:
        logger.error(f"❌ Failed to sync Data Docs via SDK: {e}")

def validate_df(df: DataFrame, suite_name: str):
    """
    Validates a Spark DataFrame using a Great Expectations suite.
    Fails the task (raises Exception) if validation fails.
    """
    logger.info(f"🔍 Starting Data Validation for suite: {suite_name}")
    
    # 1. Get Context
    try:
        gx_dir = os.path.join(ROOT_DIR, "gx_workspace")
        context = gx.get_context(project_root_dir=gx_dir)
    except Exception as e:
        logger.error(f"❌ Failed to load GX Context: {e}")
        raise
    
    # 2. Setup Spark Datasource
    ds_name = "spark_datasource"
    try:
        datasource = context.data_sources.get(ds_name)
    except:
        datasource = context.data_sources.add_spark(ds_name)
    
    # 3. Setup Data Asset & Batch Definition
    asset_name = f"asset_{suite_name}"
    try:
        asset = datasource.get_asset(asset_name)
    except:
        asset = datasource.add_dataframe_asset(name=asset_name)
    
    batch_def_name = f"batch_def_{suite_name}"
    try:
        batch_definition = asset.get_batch_definition(batch_def_name)
    except:
        batch_definition = asset.add_batch_definition_whole_dataframe(name=batch_def_name)
    
    # 4. Get Batch
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # 5. Get Expectation Suite
    try:
        suite = context.suites.get(suite_name)
    except:
        logger.warning(f"⚠️ Expectation Suite '{suite_name}' not found. Creating an empty one.")
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    
    # 6. Run Validator
    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite
    )
    
    # 7. Execute Validation
    results = validator.validate()
    
    # 8. Build Data Docs (Local)
    logger.info("📑 Building Local Data Docs...")
    context.build_data_docs()
    
    # 9. Sync to GCS
    sync_to_gcs()
    
    if not results.success:
        logger.error(f"❌ Data Quality Issue detected in suite '{suite_name}'!")
        fail_count = results.statistics['unsuccessful_expectations']
        logger.error(f"Total Failures: {fail_count}")
        raise ValueError(f"Data Quality Validation Failed: {suite_name}. Check GCP Bucket for updated reports.")
    
    logger.success(f"✅ Data Quality Passed for suite '{suite_name}'.")
    return results
