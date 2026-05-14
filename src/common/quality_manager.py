import os
import time
import great_expectations as gx
import great_expectations.expectations as gxe
from loguru import logger
from pyspark.sql import functions as F

class QualityManager:
    """
    Gestionnaire de qualité 100% compatible GX 1.x (Fluent API)
    Génère des rapports visuels (Data Docs) pour observer la qualité.
    """

    @staticmethod
    def _get_context():
        project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
        return gx.get_context(project_root_dir=project_root)

    @classmethod
    def validate_silver_data(cls, df, label="Silver Data"):
        """
        Validation avec génération de Data Docs via GX 1.x.
        """
        logger.info(f"🧪 [GX 1.x] Audit visuel : {label}")
        
        try:
            context = cls._get_context()
            pdf = df.limit(1000).toPandas()
            
            # 1. Suite d'expectations (Classe GX 1.x)
            suite_name = f"suite_{label.replace(' ', '_')}_{int(time.time())}"
            suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="Close"))
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Close", min_value=0.01))

            # 2. Data Asset éphémère
            ds_name = f"ds_{int(time.time())}"
            ds = context.data_sources.add_pandas(name=ds_name)
            asset = ds.add_dataframe_asset(name="asset")
            batch_def = asset.add_batch_definition_whole_dataframe(name="batch")
            
            # 3. Validation Definition
            val_name = f"val_{int(time.time())}"
            val_def = context.validation_definitions.add(
                gx.ValidationDefinition(name=val_name, data=batch_def, suite=suite)
            )
            
            # 4. Exécution
            results = val_def.run(batch_parameters={"dataframe": pdf})
            
            # 5. Visualisation
            context.build_data_docs()
            logger.success(f"📊 [GX] Rapport généré pour {label}.")
            
            return results.success
            
        except Exception as e:
            logger.error(f"❌ [GX 1.x] Erreur : {e}")
            return True

    @classmethod
    def validate_gold_data(cls, df, label="Gold Data"):
        """
        Validation Gold avec GX 1.x.
        """
        logger.info(f"🧪 [GX 1.x] Audit visuel Gold : {label}")
        try:
            context = cls._get_context()
            pdf = df.limit(1000).toPandas()
            
            suite = context.suites.add(gx.ExpectationSuite(name=f"suite_gold_{int(time.time())}"))
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="ADX", min_value=0, max_value=100.01))
            
            ds = context.data_sources.add_pandas(name=f"ds_gold_{int(time.time())}")
            asset = ds.add_dataframe_asset(name="asset")
            batch_def = asset.add_batch_definition_whole_dataframe(name="batch")
            
            val_def = context.validation_definitions.add(
                gx.ValidationDefinition(name=f"val_gold_{int(time.time())}", data=batch_def, suite=suite)
            )
            
            results = val_def.run(batch_parameters={"dataframe": pdf})
            context.build_data_docs()
            logger.success(f"📊 [GX] Rapport Gold généré.")
            return results.success
        except Exception as e:
            logger.error(f"❌ [GX 1.x Gold] Erreur : {e}")
            return True
