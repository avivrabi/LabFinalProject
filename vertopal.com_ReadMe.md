---
jupyter:
  application/vnd.databricks.v1+notebook:
    computePreferences:
      hardware:
    environmentMetadata:
      environment_version: 4
    language: python
    notebookMetadata:
      pythonIndentUnit: 4
    notebookName: ReadMe
    widgets: {}
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"7d093648-9e66-4a10-82d6-1ed1e41cf536\",\"showTitle\":true,\"tableResultSettingsMap\":{},\"title\":\"Cell 5\"}"}
# Running Instructions
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"1a649ac9-39fc-47e1-9684-cc2fb76f3eb5\",\"showTitle\":true,\"tableResultSettingsMap\":{},\"title\":\"Cell 6\"}"}
## Data Location

All project data is stored in the **data submissions container** under
`submissions/Ron_Aviv_Naomi`.

## Quick Start - Main Experiment Only (Suggested)

To run the main experiment with pre-processed data:

1.  Download the Notebook:
    `5. Main Experiment - Local model training and evaluation`
2.  Download the `utils.py` file to your Databricks workspace directory
3.  Run the notebook:
    **`5. Main Experiment - Local model training and evaluation`**

## Full Replication - Complete Pipeline

To replicate the entire data processing pipeline:

1.  Download all code files from the git repository (5 notebooks +
    utils.py)
2.  Run notebooks **sequentially in order 1-5** (start notebook i+1 only
    after i finish):
    -   Notebook 1: Data Collection and Integration
    -   Notebook 2: Data Cleaning and Feature Selection
    -   Notebook 3: Paris specific features
    -   Notebook 4: Training The Global Model (see note below)
    -   Notebook 5: Main Experiment - Local model training and
        evaluation

-   To run the experiment with your file, you\'ll need to skip cell 5 in
    the 5th notebook
-   This will generate the data files under
    `/FileStore/tables/paris_project/` and save the original deployment
    files untouched in the azure blob container.
-   Note that this run will take a few hours (not a reasonable timeframe
    as mentioned in the instructions)

## Important Notes

-   The `utils.py` file must be in the same directory as the experiment
    notebooks

-   ## Since we could not load sas tokens to git, use the keys in the report submitted in moodle (the one in this repo doesn\'t contain them), there are two different sas tokens for one for the original airbnb data and one for the data saved in the blob container

    Enjoy Paris! Bon voyage 🗼
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"5eb3393f-0615-4941-85f3-eda90378dac6\",\"showTitle\":false,\"tableResultSettingsMap\":{},\"title\":\"\"}"}
# Data Flow
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"39c16c6f-a558-4730-a96b-d4d174a58048\",\"showTitle\":true,\"tableResultSettingsMap\":{},\"title\":\"Cell 2\"}"}
## Data Flow Architecture - Medallion Layers

This project follows a **medallion architecture** with multiple data
quality layers:

### 🥉 **Bronze Layer** {#-bronze-layer}

Raw data ingestion zone - stores data in its original format as received
from source systems. Minimal transformations, preserving data lineage.

### 🥈 **Silver Layer** {#-silver-layer}

Cleaned and validated data - applies data quality rules, deduplication,
standardization, and basic transformations on target columns. Data is
split into training and test sets.

### 🥇 **Gold Layer** {#-gold-layer}

Aggregated and enriched data - contains business-level aggregations,
metrics, and feature engineering for analytics and ML. Includes
Paris-specific geographic features (metros, monuments).

### 💎 **Diamond Layer** {#-diamond-layer}

Production-ready datasets - contains local training and test sets from
Gold layer with global model predictions column. Ready for final
experiment runs and deployment.
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"63139975-5d9e-4ebd-805c-b70fa0b4100f\",\"showTitle\":false,\"tableResultSettingsMap\":{},\"title\":\"\"}"}
``` python
def get_parquet_file_count_and_size(path_str):
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
            return 0,0
        
        file_sizes = [f.size for f in parquet_files] 
        total_size_bytes = sum(file_sizes)
        
        return file_count, total_size_bytes

    except Exception as e:
        print(f"Error accessing {path_str}: {e}")
        return 0,0
```
:::

::: {.cell .markdown application/vnd.databricks.v1+cell="{\"cellMetadata\":{},\"inputWidgets\":{},\"nuid\":\"850ce3ef-fa56-4a21-a59b-489c1894f0f8\",\"showTitle\":false,\"tableResultSettingsMap\":{},\"title\":\"\"}"}
Below is the structure that will be excepted after running the full (5
notebooks) experiment:
:::

::: {.cell .code execution_count="0" application/vnd.databricks.v1+cell="{\"cellMetadata\":{\"byteLimit\":2048000,\"rowLimit\":10000},\"inputWidgets\":{},\"nuid\":\"e5001ed5-cc40-45f0-90b5-582236772ea1\",\"showTitle\":true,\"tableResultSettingsMap\":{},\"title\":\"Verify deployment directory structure\"}"}
``` python
# Verify the final deployment structure
project_base = "/FileStore/tables/paris_project"

print("\n" + "="*60)
print("DEPLOYMENT DIRECTORY STRUCTURE")
print("="*60 + "\n")

# Check each layer
layers_to_check = ['bronze', 'silver', 'gold', 'diamond']

for layer in layers_to_check:
    layer_path = f"{project_base}/{layer}"
    try:
        items = dbutils.fs.ls(layer_path)
        print(f"\n📁 {layer.upper()} ({len(items)} items):")
        for item in items:
            n_files, size = get_parquet_file_count_and_size(item.path)
            size_mb = size / (1024 * 1024)
            print(f"  • {item.name} - {n_files} files, {size_mb:.2f} MB")
    except Exception as e:
        print(f"\n⚠️ {layer.upper()}: Not found or empty")

print("\n" + "="*60)
```

::: {.output .stream .stdout}

    ============================================================
    DEPLOYMENT DIRECTORY STRUCTURE
    ============================================================


    📁 BRONZE (4 items):
      • global_org_df.parquet/ - 8 files, 13932.58 MB
      • local_org_df.parquet/ - 1 files, 54.36 MB
      • paris_metros_org.parquet/ - 1 files, 0.01 MB
      • paris_monuments_org.parquet/ - 1 files, 0.00 MB

    📁 SILVER (3 items):
      • global_train_v2.parquet/ - 8 files, 9026.16 MB
      • local_train_pool_v2.parquet/ - 1 files, 30.15 MB
      • test_set_v2.parquet/ - 1 files, 7.44 MB

    📁 GOLD (5 items):
      • global_train_features_v4.parquet/ - 4 files, 53.04 MB
      • local_train_pool_v4.parquet/ - 1 files, 1.77 MB
      • local_train_pool_v5_with_paris_features.parquet/ - 1 files, 6.39 MB
      • test_set_v4.parquet/ - 1 files, 0.51 MB
      • test_set_v5_with_paris_features.parquet/ - 1 files, 1.77 MB

    📁 DIAMOND (2 items):
      • local_train_with_global_pred_v7.parquet/ - 1 files, 8.13 MB
      • test_set_with_global_pred_v7.parquet/ - 1 files, 2.16 MB

    ============================================================
:::
:::
