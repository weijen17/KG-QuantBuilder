"""Tests for agents"""
import pytest
from unittest.mock import Mock, patch
from src.agents import main_workflow1,main_workflow2

def test_research_agent_initialization():
    """Test ResearchAgent can be initialized"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        agent = main_workflow1()
        assert agent is not None
        assert agent.llm is not None

def test_analyst_agent_initialization():
    """Test AnalystAgent can be initialized"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        agent = main_workflow2()
        assert agent is not None
        assert agent.llm is not None
