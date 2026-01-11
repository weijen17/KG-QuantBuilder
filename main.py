"""
Main entry point for the LLM Report System
"""

import os
from dotenv import load_dotenv
import argparse
from src.config.settings import settings
from src.agents import reader_module,extraction_module,preprocess_module,clustering_module,cluster_naming_module,KG_Building_module
import pandas as pd

RESULT_NAME = settings.RESULT_NAME
PARTIAL_DIR = settings.PARTIAL_DIR
RESULT_DIR = settings.RESULT_DIR
PARTIAL_DIR = settings.PARTIAL_DIR
RESULT_NAME = settings.RESULT_NAME

### Can consider agglo clustering, and then repeatedly merge cluster with same entity.
### Can add in eval to ensure coverage.
if __name__ =='__main__':
    l_data = reader_module()
    l_extract = extraction_module(l_data)
    print(l_extract)
    df_extract_flatten = preprocess_module(l_extract)
    df_extract_flatten.sort_values(by=['freq'],ascending=False)
    df_cluster = clustering_module(df_extract_flatten)
    df_cluster_w_name, df_cluster_group = cluster_naming_module(df_cluster)
    final_kg = KG_Building_module(df_cluster_group)
    print(final_kg)

    with pd.ExcelWriter(PARTIAL_DIR / '聚类结果_v1.1test.xlsx') as writer:
        df_cluster_w_name.to_excel(writer, sheet_name='聚类-原生数据')
        df_cluster_group.to_excel(writer, sheet_name='聚类-汇总结果')

    try:
        with open(RESULT_NAME, 'w') as f:
            f.write(final_kg)
        print("Successfully wrote to note.txt in 'w' mode.")
    except IOError as e:
        print(f"An error occurred: {e}")