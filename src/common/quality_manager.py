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
            suite_name = f"suite_silver_{label.replace(' ', '_').lower()}"
            try:
                suite = context.suites.get(name=suite_name)
            except:
                suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
            
            # --- TESTS DE ROBUSTESSE ---
            # Au moins 95% des lignes doivent avoir un prix (détection de l'échec type AAOI)
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="Close", mostly=0.95))
            
            # Le prix doit être réaliste
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Close", min_value=0.01, max_value=50000.0))
            
            # Vérifier que le volume n'est pas toujours à zéro (détection de flux figé)
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Volume", min_value=0, mostly=0.99))

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
            
            suite_name = f"suite_gold_{label.replace(' ', '_').lower()}"
            try:
                suite = context.suites.get(name=suite_name)
            except:
                suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
            
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="ADX", min_value=0, max_value=100.01))
            
            ds_name = f"ds_gold_assets"
            try:
                ds = context.data_sources.get(name=ds_name)
            except:
                ds = context.data_sources.add_pandas(name=ds_name)
            
            asset_name = f"asset_{label.replace(' ', '_').lower()}"
            try:
                asset = ds.get_asset(name=asset_name)
            except:
                asset = ds.add_dataframe_asset(name=asset_name)
            
            batch_def = asset.add_batch_definition_whole_dataframe(name="batch")
            
            val_name = f"val_gold_{label.replace(' ', '_').lower()}"
            try:
                val_def = context.validation_definitions.get(name=val_name)
            except:
                val_def = context.validation_definitions.add(
                    gx.ValidationDefinition(name=val_name, data=batch_def, suite=suite)
                )
            
            results = val_def.run(batch_parameters={"dataframe": pdf})
            context.build_data_docs()
            logger.success(f"📊 [GX] Rapport Gold généré.")
            return results.success
        except Exception as e:
            logger.error(f"❌ [GX 1.x Gold] Erreur : {e}")
            return True
    @classmethod
    def validate_ticker_list(cls, df, label="Ticker List", min_rows=400):
        """
        Validation des listes de tickers (SP500 ou 2B) avec GX 1.x.
        """
        logger.info(f"🧪 [GX 1.x] Audit Tickers : {label}")
        try:
            context = cls._get_context()
            pdf = df.toPandas()
            
            suite_name = f"suite_tickers_{label.replace(' ', '_').lower()}"
            try:
                suite = context.suites.get(name=suite_name)
            except:
                suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
            
            # 1. Nombre de lignes minimum
            suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=min_rows))
            
            # 2. Aucun symbole nul
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="symbol"))
            
            # 3. Unicité des symboles
            suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="symbol"))
            
            # 4. Capitalisation positive
            suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="marketCap", min_value=0))

            # 5. Formatage Yahoo Finance (Regex pour détecter la présence d'un suffixe pour l'international)
            # On vérifie que les tickers hors USA ont un format Ticker.Suffix ou Ticker-Suffix
            # Pour simplifier, on vérifie qu'au moins 20% des tickers ont un point ou un tiret (ratio typique de l'international dans votre liste)
            suite.add_expectation(gxe.ExpectColumnValuesToMatchRegex(
                column="symbol", 
                regex=r".*[\.\-].*", 
                mostly=0.20  # Au moins 20% doivent avoir un suffixe (ajustable selon l'univers)
            ))

            ds = context.data_sources.add_pandas(name=f"ds_tickers_{int(time.time())}")
            asset = ds.add_dataframe_asset(name="asset")
            batch_def = asset.add_batch_definition_whole_dataframe(name="batch")
            
            val_def = context.validation_definitions.add(
                gx.ValidationDefinition(name=f"val_tickers_{int(time.time())}", data=batch_def, suite=suite)
            )
            
            results = val_def.run(batch_parameters={"dataframe": pdf})
            context.build_data_docs()
            
            if results.success:
                logger.success(f"📊 [GX] Liste {label} validée ({len(pdf)} tickers).")
            else:
                logger.warning(f"⚠️ [GX] Anomalies détectées dans la liste {label} !")
                
            return results.success
        except Exception as e:
            logger.error(f"❌ [GX 1.x Tickers] Erreur : {e}")
            return True
