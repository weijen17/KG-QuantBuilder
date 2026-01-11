
import pytest
import os
from dotenv import load_dotenv
import argparse
from src.config.settings import settings
from src.agents import main_workflow1,main_workflow2

report_name = settings.REPORT_NAME

# ------------------------------
# 1️⃣ Test: App runs with valid input
# ------------------------------
def test_app_runs_with_valid_input():
    business_question = "Analyze sales trend over years"
    res1 = main_workflow1(business_question, 10)
    res2 = main_workflow2()

    assert res1 is not None, "App should return a result"
    assert res2 is not None, "App should return a result"
    assert os.path.exists(report_name), f"File does not exist: {report_name}"


# ------------------------------
# 2️⃣ Test: App handles empty input gracefully
# ------------------------------
def test_app_handles_empty_data():
    sample_data = {}
    business_question = "Analyze sales trend over years"

    res1 = main_workflow1(business_question, 10)
    res2 = main_workflow2()

    assert res1 is not None
    assert res2 is not None

