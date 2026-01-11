
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
import os
import pandas as pd
from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import List

from src.config.settings import settings


REPORT_NAME = settings.REPORT_NAME
save_path__json = settings.JSON_DIR

llm = init_chat_model(model=settings.MODEL_NAME, model_provider=settings.MODEL_PROVIDER, temperature=settings.TEMPERATURE,top_p=settings.TOP_P)

########################################################################################################################

eval_prompt__conciseness = f'''
## Role
You are an expert reviewer of data analytics report.

## Task
You are evaluating a report for conciseness. A concise report delivers key information clearly and directly without unnecessary elaboration, redundancy, or verbosity.
Evaluate the following report on a scale of 1-5 for conciseness:
1 = Very verbose, repetitive, and unnecessarily long
2 = Somewhat wordy with noticeable redundancy
3 = Moderately concise with some unnecessary details
4 = Concise with minimal excess content
5 = Highly concise, every sentence adds value

## Instruction
1. A score (1-5)
2. Brief justification for the score
3. Output the score and justification in a list, the first element is being the score and the second element being the justification. Do not output anything else.
4. Do not output anything else, apart from the list
'''


eval_prompt__coherence = f'''
## Role
You are an expert reviewer of data analytics report.

## Task
You are evaluating a report for coherence. A coherent report has logical flow, clear connections between ideas, consistent structure, and unified narrative that is easy to follow.
Evaluate the following report on a scale of 1-5 for coherence:
1 = Disjointed, confusing structure with no logical flow
2 = Poor organization with weak connections between sections
3 = Adequately organized but some transitions are unclear
4 = Well-organized with clear logical flow and good transitions
5 = Exceptionally coherent with seamless flow and unified narrative

## Instruction
1. A score (1-5)
2. Brief justification for the score
3. Output the score and justification in a list, the first element is being the score and the second element being the justification. Do not output anything else.
4. Do not output anything else, apart from the list
'''


eval_prompt__coverage = f'''
## Role
You are an expert reviewer of data analytics report.

## Task
You are evaluating a report for coverage of business questions. A report with sufficient coverage thoroughly addresses all aspects of the business question, provides relevant insights, and doesn't leave key areas unexplored.
Evaluate the following report on a scale of 1-5 for coverage:
1 = Fails to address the business question or missing major components
2 = Addresses some aspects but significant gaps remain
3 = Covers main points but lacks depth or misses some dimensions
4 = Comprehensive coverage with minor gaps
5 = Complete and thorough coverage of all relevant aspects

## Instruction
1. A score (1-5)
2. Brief justification for the score
3. Output the score and justification in a list, the first element is being the score and the second element being the justification. 
4. Do not output anything else, apart from the list
'''

eval_prompt__faithfulness = f'''
## Role
You are an expert reviewer of data analytics report.

## Task
You are evaluating a report for data faithfulness. A faithful report accurately represents the data without misinterpretation, exaggeration, or unsupported claims.
Evaluate the following report on a scale of 1-5 for faithfulness to data:
1 = Significant misrepresentation or claims contradicting the data
2 = Multiple inaccuracies or unsupported claims
3 = Generally accurate but some interpretation issues or overgeneralizations
4 = Faithful to data with minor interpretation concerns
5 = Completely accurate and faithful to the data

## Instruction
1. A score (1-5)
2. Brief justification for the score
3. Output the score and justification in a list, the first element is being the score and the second element being the justification. Do not output anything else.
4. Do not output anything else, apart from the list
'''

########################################################################################################################

l_metric=['coherence','conciseness','coverage','faithfulness']
def report_evaluation(metric,human_input):
    if metric =='coherence':
        eval_prompt =eval_prompt__coherence
    elif metric =='conciseness':
        eval_prompt = eval_prompt__conciseness
    elif metric =='coverage':
        eval_prompt = eval_prompt__coverage
    elif metric == 'faithfulness':
        eval_prompt = eval_prompt__faithfulness
    else:
        eval_prompt = None

    if metric in l_metric:
        response = llm.invoke([SystemMessage(content=eval_prompt),
                                  HumanMessage(content=human_input)])
        res = response.content
    else:
        res = None
    return res


def finding_evaluation(metric,human_input):
    if metric =='coherence':
        eval_prompt =eval_prompt__coherence
    elif metric =='conciseness':
        eval_prompt = eval_prompt__conciseness
    elif metric =='coverage':
        eval_prompt = eval_prompt__coverage
    elif metric == 'faithfulness':
        eval_prompt = eval_prompt__faithfulness
    else:
        eval_prompt = None

    if metric in l_metric:
        response = llm.invoke([SystemMessage(content=eval_prompt),
                                  HumanMessage(content=human_input)])
        res = response.content
    else:
        res = None
    return res



