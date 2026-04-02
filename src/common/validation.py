import os
import sys
import great_expectations as gx
from pyspark.sql import DataFrame
from loguru import logger

# Correct path for GE context
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def validate_spark_df(df: DataFrame, suite_name: str):
    """
    Validates a Spark DataFrame using a Great Expectations suite.
    Fails the task (raises Exception) if validation fails.
    Configured to build Data Docs on GCS on completion.
    """
    logger.info(f"🔍 Starting Data Validation for suite: {suite_name}")
    
    # 1. Get Context (GE 1.x)
    try:
        # Avoid shadowing by pointing to the renamed workspace folder
        gx_dir = os.path.join(ROOT_DIR, "gx_workspace")
        context = gx.get_context(project_root_dir=gx_dir)
    except Exception as e:
        logger.error(f"❌ Failed to load GX Context: {e}")
        raise
    
    # 2. Setup Spark Datasource (Fluent API)
    ds_name = "spark_datasource"
    try:
        datasource = context.data_sources.get(ds_name)
    except:
        datasource = context.data_sources.add_spark(ds_name)
    
    # 3. Setup Data Asset & Batch Definition (GX 1.x pattern)
    asset_name = f"asset_{suite_name}"
    try:
        asset = datasource.get_asset(asset_name)
    except:
        asset = datasource.add_dataframe_asset(name=asset_name)
    
    # Add batch definition if not exists
    batch_def_name = f"batch_def_{suite_name}"
    try:
        batch_definition = asset.get_batch_definition(batch_def_name)
    except:
        batch_definition = asset.add_batch_definition_whole_dataframe(name=batch_def_name)
    
    # 4. Get Batch (Passing the actual Spark DataFrame here)
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # 5. Get Expectation Suite
    try:
        suite = context.suites.get(suite_name)
    except:
        logger.warning(f"⚠️ Expectation Suite '{suite_name}' not found. Creating an empty one.")
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    
    # 6. Run Validator using the Batch
    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite
    )
    
    # 7. Execute Validation
    results = validator.validate()
    
    # 8. Always build Data Docs for traceability
    logger.info("📑 Building Data Docs on GCS...")
    context.build_data_docs()
    
    if not results.success:
        logger.error(f"❌ Data Quality Issue detected in suite '{suite_name}'!")
        # Find the specific failures
        fail_count = results.statistics['unsuccessful_expectations']
        logger.error(f"Total Failures: {fail_count}")
        
        # Raise exception to kill Airflow DAG
        raise ValueError(f"Data Quality Validation Failed: {suite_name}. Check Data Docs on GCS for details.")
    
    logger.success(f"✅ Data Quality Passed for suite '{suite_name}'.")
    return results
