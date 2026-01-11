# KG-QuantBuilder

- Build entities hierarchical labeling system (KG Framework) based on entities extracted from text corpora

## Features

- **Entity Extraction**: LLM performs entities extraction from text corpora.
- **Embedding Clustering**: Perform clustering based on entities embeddings.
- **KG Framework**: LLM generates entities hierarchical labeling framework based on clustering result.

## Prerequisites

- Python 3.12+``
- Docker and Docker Compose (for containerized deployment)
- OpenAI API keys (Default Model Spec: GPT-4o)

## Setup

### Local Setup

1. Clone the repository:
```bash
git clone 
cd KG-Quantbuilder
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. Run the application:
```bash
python main.py
```

### Docker Setup

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
`````

2. Build and run with Docker Compose:
```bash
docker-compose run --rm KG-Framework-Generator

or

docker-compose up --build
```

## Main Application Usage

1. Place raw_data in **input** directory. 
2. Amend env file accordingly.
3. Amend prompts in **src/assets** directory (such as 效果、场景、材料、成分）.
3. After executing main.py, the generated final result will be in **artifact/result** directory.


## Module Workflow

```
**Workflow1**
┌─────────────┐
│   Input     │──► Text corpora in excel file
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Extraction │──► Extract entities
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Clustering │──► Perform clustering on embedding (m3e-base)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ KG Framework│──► Generate KG hierarchical labeling framework based on clustering result
└─────────────┘
```


# File System Structure
```
project/
├── artifact/       # Generated outputs and intermediate artifacts
│   ├── intermediate/   # Saved intermediate artifacts
│   ├── result/     # **Final KG Framework result**
├── eval/           # Evaluation test scripts (not usable, pending)
│   ├── eval_result/# Evaluation result files (not usable, pending)
├── input/          # Input data files
│   ├── raw_data/   # **Raw Data**
├── logs/           # Logs
├── misc/           # Miscellaneous files
├── src/            # Source code
│   ├── agents/     # Primary Agents Source Code
│   ├── assets/     # Static resources (prompts, templates)
│   ├── configs/    # Configuration files and environmental variables
│   └── tools/      # Utility functions
└── tests/          # Test suites ongoing (boilerplate code)
```

## Pending

1. Use agglomerative clustering, and uses LLM to merge similar cluster repetitively before generating final result.