
 df
jsדשדגs  sdasf
rom pyspark.sql.functions import broadcast, col, expm1, abs as spark_abs
from pyspark.sql import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


def save_data_set(data,
                  save_path: str,
                  num_pages: int,
                  base_path: str
                  ):
    """
    Saves a Spark DataFrame to a Parquet file in the specified path.
    """
    assert(isinstance(data, DataFrame))
    print(f"Saving data set to to {base_path}/{save_path} ...")
    data.coalesce(num_pages).write.mode("overwrite").parquet(f"{base_path}/{save_path}")
    print(f"✅ Data set saved to {base_path}/{save_path}.")


def load_data_and_check_leak(spark,
                             base_path: str,
                             global_path: str = "global_train_v2.parquet",
                             local_path: str = "local_train_pool_v2.parquet",
                             test_path: str = "test_set_v2.parquet",
                             test_needed: bool = False):     # TODO: Add checks and versatile behaviour for not all 3 datasets required!
    
    # ==============================================================================
    # LOAD DATA ARTIFACTS (Gold Layer)
    # ==============================================================================

    if base_path:
        print(f"Loading data from: {base_path}...\n")
        global_path = f"{base_path}/{global_path}" if global_path else ""
        local_path = f"{base_path}/{local_path}" if local_path else ""
        test_path = f"{base_path}/{test_path}" if test_path else ""

    ret_data = {}
    # 1. Load Global Train
    if global_path:
        global_train_df = spark.read.parquet(global_path)
        global_size = global_train_df.count()
        print(f"✅ Global Train Count: {global_size:,}")
        ret_data['global_train'] = global_train_df

    # 2. Load Local Train Pool
    if local_path:
        local_train_df = spark.read.parquet(local_path)
        local_size = local_train_df.count()
        print(f"✅ Local Train Count:  {local_size:,}")
        ret_data['local_train'] = local_train_df

    # 3. Load Test Set
    test_df = spark.read.parquet(test_path)
    test_size = test_df.count()
    print(f"✅ Test Set Count:      {test_size:,}")
    if test_needed:
        ret_data['test'] = test_df

    # ==============================================================================
    # VERIFICATION (Optimized with Broadcast)
    # ==============================================================================

    # Since test_df is small, we send it to the workers instead of shuffling the train data.
    local_leakage = local_train_df.join(broadcast(test_df), on="id", how="inner").count() if local_path else 0
    global_leakage = global_train_df.join(broadcast(test_df), on="id", how="inner").count() if global_path else 0

    if  local_leakage > 0 or global_leakage > 0:
        err_msg = ""
        if local_leakage > 0:
            err_msg += f"CRITICAL: Data Leakage In Local Train Set: {local_leakage:,} test records found.\n"
        if global_leakage > 0:
            err_msg += f"CRITICAL: Data Leakage In Global Train Set: {global_leakage:,} test records found.\n"
        raise Exception(err_msg)
    else:
        print("✅ No Data Leakage Found (Verified via Broadcast Join)")

    return ret_data



def analyze_storage_layout(dbutils,path_str):
    """
    Analyzes the storage layout of a given path and
    prints the number of parquet files and their average size in MB.
    """
    try:
        files = dbutils.fs.ls(path_str)
        parquet_files = [f for f in files if f.name.endswith(".parquet")]
        file_count = len(parquet_files)
        
        if file_count == 0:
            print(f"Path: {path_str} -> No parquet files found directly (might be nested directories).")
            return
        
        file_sizes = [f.size for f in parquet_files] 
        total_size_bytes = sum(file_sizes)
        avg_size_mb = (total_size_bytes / file_count) / (1024 * 1024)
        max_size_mb = max(file_sizes) / (1024 * 1024)
        
        print(f"--- Analysis for: {path_str} ---")
        print(f"Total Parquet Files: {file_count}")
        print(f"Total Size: {total_size_bytes / (1024*1024*1024):.2f} GB")
        print(f"Average File Size: {avg_size_mb:.2f} MB")
        print(f"Max File Size: {max_size_mb:.2f} MB")

        if file_count > 100 and avg_size_mb < 10:
            print("CRITICAL WARNING: Small Files Problem Detected! This causes severe latency.")
        if max_size_mb > 3 * avg_size_mb:
            print("WARNING: Large Max File Size Detected! the data is not efficiently distributed over the nodes. consider repartitioning instead colcase.")
        elif avg_size_mb > 100:
             print("GOOD LAYOUT: Files are large and optimized for reading.")

    except Exception as e:
        print(f"Error accessing {path_str}: {e}")


    # --- Helper Function for Visualizations ---
def log_visualizations(predictions_df, final_model, chosen_model_name, title_prefix=""):
    """
    Samples the predictions and logs plots to MLflow.
    Using sampling to avoid OOM on the driver.
    """
    # --- NEW: Feature Importance Plot ---
    # This only works for Tree-based models (GBT/RF)
    # We need to extract the 'feature_importances_' attribute from the last stage of the pipeline
    
    trained_estimator = final_model.stages[-1]
    
    if hasattr(trained_estimator, 'featureImportances'):
        print("📊 Generating Feature Importance Plot...")
        
        # 1. Extract Feature Names from the DataFrame Schema (Not the Assembler object)
        # The metadata is hidden inside the 'features' column definition
        try:
            # We access the schema of the PREDICTIONS dataframe
            feature_attr = predictions_df.schema['features'].metadata['ml_attr']['attrs']
            
            features_list = []
            # Spark separates attributes by type, we need to collect them all
            if 'numeric' in feature_attr: features_list += feature_attr['numeric']
            if 'binary' in feature_attr: features_list += feature_attr['binary']
            
            # Sort by index to match the model's output vector (Critical!)
            features_list.sort(key=lambda x: x['idx'])
            feature_names = [x['name'] for x in features_list]
            
        except (KeyError, AttributeError) as e:
            print("⚠️ Warning: Could not extract specific feature names from metadata. Using generic indices.")
            print(f"Exeption catched: {e}")
            # Fallback: Just create Feature_0, Feature_1, etc.
            feature_names = [f"Feature_{i}" for i in range(len(trained_estimator.featureImportances))]

        # 2. Extract Importances
        importances = trained_estimator.featureImportances.toArray()
        
        # 3. Create Pandas DF for plotting
        # Ensure lengths match (sometimes metadata is tricky)
        min_len = min(len(feature_names), len(importances))
        
        fi_df = pd.DataFrame({
            'Feature': feature_names[:min_len], 
            'Importance': importances[:min_len]
        })
        
        # Top 10 Features
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        
        # 4. Plot
        fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
        sns.barplot(data=fi_df, x='Importance', y='Feature', color='#002855', ax=ax_fi)
        ax_fi.set_title(f'Top 10 Drivers of Price ({chosen_model_name})')
        ax_fi.set_xlabel('Relative Importance')
        
        mlflow.log_figure(fig_fi, "feature_importance.png")
        plt.show() # Display in notebook
        
    else:
        print("ℹ️ Selected model does not support Feature Importance (e.g. Linear Regression).")

    # 1. Sample ~10k rows for plotting (Lightweight)
    pdf = predictions_df.selectExpr(
        "expm1(log_price) as actual", 
        "expm1(prediction) as pred"
    ).sample(False, 0.05, seed=42).limit(10000).toPandas()
    
    # 2. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pdf['actual'], y=pdf['pred'], alpha=0.3, ax=ax, color='#2c3e50')
    
    # Perfect prediction line
    max_val = max(pdf['actual'].max(), pdf['pred'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_title(f'{title_prefix} Actual vs Predicted Prices')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_xlim(0, 1000) # Focus on the main distribution
    ax.set_ylim(0, 1000)
    ax.legend()
    
    # Save to MLflow
    mlflow.log_figure(fig, f"{title_prefix}_actual_vs_pred.png")
    plt.close(fig)
    
    # 3. Residual Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    residuals = pdf['actual'] - pdf['pred']
    sns.histplot(residuals, bins=50, kde=True, ax=ax2, color='#e74c3c')
    ax2.set_title(f'{title_prefix} Residuals Distribution (Actual - Pred)')
    ax2.set_xlabel('Error ($)')
    ax2.set_xlim(-200, 200)
    
    mlflow.log_figure(fig2, f"{title_prefix}_residuals_dist.png")
    plt.close(fig2)