# quoting-agent

A crane quoting agent built with Google's Agent Development Kit (ADK)
Agent generated with [`googleCloudPlatform/agent-starter-pack`](https://github.com/GoogleCloudPlatform/agent-starter-pack) version `0.4.4`

## Project Structure

This project is organized as follows:

```
quoting-agent/
├── app/                 # Core application code
│   ├── agent.py         # Main agent logic
│   ├── agent_engine_app.py # Agent Engine application logic
│   └── utils/           # Utility functions and helpers
├   └── data/            # Input email json file and currency rates file.
├── deployment/          # Infrastructure and deployment scripts
├── notebooks/           # Jupyter notebooks for prototyping and evaluation
├── tests/               # Unit, integration, and load tests
├── adk_app_deploy.ipynb # Notebook to deploy the quoting agent to Agent Engine and register with Agentspace
├── Makefile             # Makefile for common commands
└── pyproject.toml       # Project dependencies and configuration

```


## Setup 

Install required packages and launch the local development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Create a GCS bucket, and two folders: 'data/input' and 'data/output', 
- upload ./quoting-agent/app/data/emails.json to gcs bucket under ./data/input folder.
- Note each time you send a quote a customer, the quote pdf file is uploaded to ./data/output folder.
- You need go to ./quoting-agent/app/config.py and add your GCP project ID location, GCS bucket.


## Local Testing

```bash
cd ./quoting-agent/
adk web --port 8501
```

## Deploy to Agent Engine and Register with Agentspace

run the notebook ./quoting-agent/adk_app_deploy.ipynb
