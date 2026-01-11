
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
import os
import pandas as pd
from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import List
import logging
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.assets.prompts import system_prompt__effect_extraction,system_prompt__cluster_naming,system_prompt__KG_building

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

LOG_FILE = os.path.join(settings.LOG_DIR, "app.log")
# Configure logging
logging.basicConfig(
    level=logging.INFO,                           # logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),           # save to file
        logging.StreamHandler()                  # also print to console
    ]
)

llm = init_chat_model(model=settings.MODEL_NAME, model_provider=settings.MODEL_PROVIDER, temperature=settings.TEMPERATURE,top_p=settings.TOP_P)

RESULT_NAME = settings.RESULT_NAME
INPUT_FILENAME = settings.INPUT_NAME
TEXT_COL = settings.TEXT_COL
FREQ_UB = int(settings.FREQ_UB)

########################################################################################################################

def reader_module():
    logging.info(f"Reader Module commencing ...")
    # df = pd.read_excel(INPUT_FILENAME).sample(100, random_state=1)
    df = pd.read_excel(INPUT_FILENAME)
    df[TEXT_COL] = df[TEXT_COL].astype('str')
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x:x[:500])
    l_data = df[TEXT_COL].values.tolist()
    logging.info(f"Reader Module completed...")
    return l_data

def extraction_module(l_data):
    logging.info(f"Extraction Module commencing ...")
    l_extract = []
    for _ in l_data:
        try:
            task_response = llm.invoke([SystemMessage(content=system_prompt__effect_extraction),
                                        HumanMessage(content=_)])
            task_res = task_response.content
        except:
            task_res = None
        l_extract.append(task_res)
    logging.info(f"Extraction Module completed ...")
    return l_extract


def preprocess_module(l_extract):
    logging.info(f"Preprocess Module commencing ...")
    l_extract_dedup =  [set(i.split('、')) for i in l_extract]
    l_extract_flatten = [i for j in l_extract_dedup for i in j]
    df_extract_flatten = pd.DataFrame(l_extract_flatten,columns=['entity'])
    df_extract_flatten['freq'] = 1
    df_extract_flatten = df_extract_flatten.groupby(['entity'])['freq'].sum().reset_index()
    df_extract_flatten.sort_values(by=['freq'], ascending=False,inplace=True)
    logging.info(f"Preprocess Module completed ...")
    return df_extract_flatten


def clustering_module(df_extract_flatten):
    logging.info(f"Clustering Module commencing ...")
    model = SentenceTransformer('moka-ai/m3e-base')

    df_extract_flatten2=df_extract_flatten[df_extract_flatten['freq']>=FREQ_UB].reset_index(drop=True)
    l_text = df_extract_flatten2['entity'].values.tolist()
    len_text = len(l_text)
    l_embeddings = []
    _div=1000
    for enum in range((len_text//_div)+1):
        print(enum)
        embeddings = model.encode(l_text[enum*_div:(enum+1)*_div])
        l_embeddings+=embeddings.tolist()

    df_emb = pd.DataFrame(l_embeddings)
    n_clusters = min(min(300,max(50,len_text//50)),len_text)
    clustering = KMeans(n_clusters=n_clusters, random_state=1, verbose=2, n_init=10).fit(l_embeddings)
    df_label = pd.DataFrame(clustering.labels_,columns=['cluster_label'])
    df_cluster = df_extract_flatten2[['entity']].join(df_label)
    df_cluster = df_cluster.sort_values(by=['cluster_label','entity'])
    df_cluster = df_cluster.reset_index(drop=True)

    df_cluster['f_seed'] = 0  # Initialize all labels to 0

    # For each group, randomly sample 10 indices and set their label to 1
    for group_name, group_df in df_cluster.groupby('cluster_label'):
        # Sample up to 10 rows (or all if less than 10)
        sample_size = min(10, len(group_df))
        sample_indices = group_df.sample(n=sample_size).index
        df_cluster.loc[sample_indices, 'f_seed'] = 1
    logging.info(f"Clustering Module completed ...")
    return df_cluster

def cluster_naming_module(df_cluster):
    logging.info(f"Cluster Naming Module commencing ...")
    l_cluster_id = df_cluster['cluster_label'].unique().tolist()

    l_cluster_name = []
    for cluster in l_cluster_id:
        df_tmp = df_cluster[(df_cluster['cluster_label']==cluster) & (df_cluster['f_seed']==1)].reset_index(drop=True)
        l_kw = '、'.join(df_tmp['entity'].values.tolist())
        try:
            task_response = llm.invoke([SystemMessage(content=system_prompt__cluster_naming),
                                        HumanMessage(content=l_kw)])
            task_res = task_response.content
        except:
            task_res = None
        l_cluster_name.append([cluster,task_res])
    df_cluster_name = pd.DataFrame(l_cluster_name,columns = ['cluster_label','cluster_name'])
    df_cluster_w_name = pd.merge(df_cluster,df_cluster_name,on='cluster_label',how='left')
    df_cluster_group = df_cluster_w_name[df_cluster_w_name['f_seed']==1].groupby(['cluster_label','cluster_name'])['entity'].apply(list).reset_index()
    df_cluster_group.rename(columns={'entity':'entity_seed_keyword'},inplace=True)
    logging.info(f"Cluster Naming Module completed ...")
    return df_cluster_w_name,df_cluster_group

def KG_Building_module(df_cluster_group):
    logging.info(f"KG Building Module commencing ...")
    l_input = [['聚类编号','聚类的类名','这个类的功效种子关键词']]+df_cluster_group.values.tolist()
    try:
        task_response = llm.invoke([SystemMessage(content=system_prompt__KG_building),
                                    HumanMessage(content=f'功效词聚类后的数据如下：\n\n{l_input}')])
        task_res = task_response.content
    except:
        task_res = None
    logging.info(f"KG Building Module completed ...")
    return task_res
