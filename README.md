# Central Bank Digital Currency Intelligence Engine

An end-to-end machine learning and graph ML intelligence platform for detecting suspicious transactions in synthetic CBDC-like payment networks. Combines traditional machine learning, Graph Neural Networks, Neo4j graph storage, MLflow experiment tracking, FastAPI inference, Docker containerization and Prometheus monitoring.

# What is a Central Bank Digital Currency (CBDC)?

A Central Bank Digital Currency (CBDC) is a digital form of a country's official currency, issued and backed by its central bank. It aims to make digital payments faster, more accessible and efficient in the future. As money becomes increasingly digital, CBDCs represent one possible direction for the future of currency, alongside cash, bank deposits, and existing digital payment systems.

However, access to real transaction data is restricted due to privacy and confidentiality. This project aims to develop and evaluate a fraud detection system using synthetic CBDC-like transaction data, combining traditional machine learning, Graph Neural Networks and Neo4j to analyse suspicious transaction patterns and wallet relationships.

**Status:** Core ML pipeline, graph-based fraud detection, API, Docker containerization and basic Prometheus monitoring implemented.

## Overview

Financial fraud can involve more than individual transactions. Suspicious behaviour may emerge from unusual transaction amounts, high-frequency transfers, account takeover patterns, and relationships between wallets.

This project explores a multi-model approach to transaction fraud detection by combining tabular machine learning with graph-based behavioural analysis.

The system uses three models:

- **Random Forest** — detects suspicious patterns in transaction features.
- **XGBoost** — provides a gradient-boosted classification baseline.
- **GraphSAGE** — learns wallet representations from transaction relationships and performs edge-level fraud classification.

All three models are integrated behind a unified FastAPI inference endpoint. The API is packaged using Docker, with Prometheus instrumentation for HTTP request metrics.

> **Data note:** The project uses synthetic CBDC-like transaction data with deliberately simulated fraud patterns. The dataset does not contain real central bank transaction records, and the reported metrics should not be interpreted as real-world banking performance.

## Architecture

```mermaid
flowchart TD
    A[Synthetic Transaction Data] --> B[Fraud Injection]
    B --> C[Feature Engineering]
    C --> D[Temporal Train-Test Split]

    D --> E[Random Forest]
    D --> F[XGBoost]

    B --> G[Neo4j Graph Storage]
    G --> H[Wallet Nodes and TRANSFER Edges]
    H --> I[PyTorch Geometric]
    I --> J[GraphSAGE Edge Classifier]

    E --> K[MLflow Experiment Tracking]
    F --> K
    J --> K

    E --> L[FastAPI Inference]
    F --> L
    J --> L
    G --> J

    L --> M[Fraud Prediction and Probability]

    M --> N[Docker Container]
    N --> O[Prometheus Instrumentation]
    O --> P[/metrics Endpoint]
    P --> Q[Prometheus Monitoring]
```

## Key Features

### 1. Synthetic Data and Fraud Simulation

- Generated a dataset of 100,000 transactions for a simulated wallet payment network.
- Implemented six fraud scenarios: account takeover, high-frequency bursts, smurfing/structuring, mule-chain behaviour, unusual spending, and circular transfers.
- Engineered behavioural and transaction features, including velocity, transaction frequency, amount deviations, device changes, location changes, and balance-related attributes.

### 2. Traditional Machine Learning

Implemented separate training pipelines for Random Forest and XGBoost.

- Chronological train-test splitting to evaluate models on later transactions.
- Numerical preprocessing and categorical encoding through scikit-learn pipelines.
- Evaluation using precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.
- MLflow tracking for experiment parameters, evaluation metrics, and model artifacts.

### 3. Graph-Based Fraud Detection

Represented the payment network as a directed graph:

- **Nodes:** Wallets.
- **Edges:** Transfers between wallets.
- **Node features:** Eight wallet-level behavioural and network statistics.
- **Edge features:** Five transaction attributes.
- **Target:** Binary transaction-level fraud classification.

Neo4j stores and queries the wallet transaction network, while PyTorch Geometric converts the graph into tensors for GraphSAGE training and inference.

The model aggregates information from neighbouring wallets, combines source and destination wallet embeddings with transaction features, and predicts the probability that a transfer is fraudulent.

The graph contains approximately 12,000 wallet nodes and 100,000 transfer relationships.

### 4. FastAPI Inference Service

Integrated all three trained models behind a single endpoint:

`POST /predict`

The API accepts a JSON transaction object and returns individual predictions and fraud probabilities from Random Forest, XGBoost, and GraphSAGE.

Example response:

```json
{
  "random_forest": {
    "prediction": "normal",
    "fraud_probability": 0.0224
  },
  "xgboost": {
    "prediction": "normal",
    "fraud_probability": 0.0
  },
  "gnn": {
    "prediction": "normal",
    "fraud_probability": 0.0448
  }
}
```

The values above are an example from a successful local API test, not general performance metrics. GNN inference retrieves wallet transaction history from Neo4j to construct the graph used for prediction.

The API also exposes a Prometheus-compatible metrics endpoint at `/metrics` for HTTP request monitoring.

### 5. Docker Containerization

Packaged the FastAPI inference service and its runtime dependencies into a Docker image.

- Created a `Dockerfile` using a Python 3.12 slim base image.
- Installed the Python dependencies required for model inference.
- Included the application source code and trained model artifacts in the image.
- Configured Uvicorn to run the API on port 8000.
- Added `.dockerignore` to exclude local environments, datasets, MLflow files and environment secrets from the Docker build context.
- Supplied Neo4j connection settings through runtime environment variables instead of embedding credentials in the image.

**Docker image:** `cbdc-intelligence-api`

**API container:** `cbdc-api`

The image was successfully built in GitHub Codespaces, and the container started the FastAPI application. The `/docs` and `/metrics` endpoints returned HTTP 200 during testing.

### 6. Prometheus Monitoring

Integrated Prometheus instrumentation into FastAPI using `prometheus-fastapi-instrumentator`.

- Exposed HTTP request metrics through `/metrics`.
- Created `prometheus.yml` to configure the `cbdc-api` scrape target.
- Configured a 15-second scrape interval.
- Ran Prometheus in a separate Docker container.
- Created a Docker network to enable communication between the API and Prometheus containers.
- Verified that HTTP request counters increased after sending requests to the API.
- Confirmed that the Prometheus target reported `UP` after resolving container connectivity issues.

**Prometheus container:** `cbdc-prometheus`

This implementation demonstrates basic application-level monitoring. Advanced alerting, model drift detection, inference latency analysis and production monitoring remain future improvements.

## Model Evaluation

Results from the synthetic dataset:

| Metric | Random Forest | XGBoost | GraphSAGE v5 |
|---|---:|---:|---:|
| Precision | | | |
| Recall | | | |
| F1-score | | | |
| ROC-AUC | | | |
| PR-AUC | | | |

**Interpretation:** These results demonstrate that the models learn the deliberately constructed signals in the synthetic dataset. The exceptionally high Random Forest and XGBoost scores are influenced by the simulated fraud patterns and should not be taken as evidence of equivalent performance on real financial transactions.

GraphSAGE uses a separate graph-based representation and is evaluated using a chronological split, with the decision threshold selected using validation data.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Traditional ML | scikit-learn, XGBoost |
| Graph ML | PyTorch, PyTorch Geometric, GraphSAGE |
| Graph database | Neo4j |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn |
| Model persistence | joblib, PyTorch checkpoints |
| Containerization | Docker |
| Monitoring | Prometheus, prometheus-fastapi-instrumentator |
| Version control | Git, GitHub |
| Development environment | GitHub Codespaces |

## Project Structure

```text
central-bank-digital-currency-intelligence-engine/
├── data/
│   ├── raw/
│   ├── processed/
│   └── inference/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── api/
│   │   ├── inference.py
│   │   ├── gnn_inference.py
│   │   └── main.py
│   ├── graph/
│   │   ├── extract_graph.py
│   │   └── train_gnn.py
│   └── test/
├── tests/
├── notebooks/
├── configs/
├── models/
├── Dockerfile
├── .dockerignore
├── prometheus.yml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Running the API Locally

### Prerequisites

- Python 3.12
- Neo4j with the project transaction graph loaded
- Trained Random Forest, XGBoost, and GraphSAGE model artifacts
- Project dependencies installed in a virtual environment

### 1. Clone the repository

```bash
git clone https://github.com/Harin22/central-bank-digital-currency-intelligence-engine.git
cd central-bank-digital-currency-intelligence-engine
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Neo4j

Create a local `.env` file in the project root:

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=cbdc
```

Ensure the Neo4j database is running and contains the expected `Wallet` nodes and `TRANSFER` relationships. Keep `.env` and credentials out of version control.

### 5. Start FastAPI

```powershell
uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Use `POST /predict` to submit a valid transaction JSON object and inspect the three model predictions.

**Note:** The API requires the trained model artifacts and graph database configuration. A fresh clone does not automatically contain the manually supplied model artifacts.

## Running with Docker

Docker and the Prometheus configuration were tested in GitHub Codespaces.

### 1. Ensure the model artifacts are available

The following files must exist locally before building the image:

```text
models/
├── random_forest.joblib
├── xgboost.joblib
└── gnn_model.pt
```

These artifacts are not automatically included in a fresh Git clone.

### 2. Build the image

```bash
docker build -t cbdc-intelligence-api .
```

### 3. Run the API container

The Neo4j connection must point to a reachable database. `127.0.0.1` inside the container refers to the container itself, not the host machine.

For example, when Neo4j runs on the Windows host using Docker Desktop:

```bash
docker run -d --name cbdc-api -p 8000:8000 \
  -e NEO4J_URI=bolt://host.docker.internal:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=your_password \
  -e NEO4J_DATABASE=cbdc \
  cbdc-intelligence-api
```

Replace the example password with your own database credential. Do not commit real credentials to GitHub.

Open the API documentation at:

http://localhost:8000/docs

**Important:** The containerized API startup and metrics endpoint were verified in Codespaces. End-to-end GNN prediction using a real, reachable Neo4j database in the container still requires validation.

## Running Prometheus Monitoring

The repository includes `prometheus.yml`, which configures Prometheus to scrape the API's `/metrics` endpoint every 15 seconds.

The API and Prometheus containers must be connected to the same Docker network for the configured `cbdc-api:8000` target to resolve.

Example setup, assuming the API container has already been started:

```bash
docker network create cbdc-net
docker network connect cbdc-net cbdc-api

docker run -d --name cbdc-prometheus \
  --network cbdc-net \
  -p 9090:9090 \
  -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus
```

If the network or container already exists, reuse it instead of creating a duplicate.

Open the Prometheus interface at:

http://localhost:9090

Verify the target health through:

http://localhost:9090/api/v1/targets

The API metrics endpoint can be checked at:

http://localhost:8000/metrics

The configuration was tested in GitHub Codespaces. Actual prediction requests still require valid model artifacts and working Neo4j connectivity.

## Future Work

- **End-to-end deployment validation:** Verify containerized predictions from all three models with a reachable Neo4j database.
- **Artifact reproducibility:** Document model artifact generation or distribution so a fresh clone can be built reliably.
- **CI/CD:** Automate testing and validation through a GitHub Actions pipeline.
- **Advanced monitoring:** Add inference latency metrics, error-rate monitoring, alerting, and data or prediction drift detection.
- **Cloud deployment:** Deploy the inference service to a cloud environment and validate its operational behaviour.

## Limitations

- The dataset is synthetic and does not represent actual CBDC transactions.
- Simulated fraud patterns may make classification easier than real-world fraud detection.
- Graph inference currently uses wallet history retrieved from Neo4j; production-scale graph serving and latency optimisation remain future work.
- The Dockerized API has been tested for startup and metrics exposure, but end-to-end GNN inference against a real Neo4j database in the container remains to be validated.
- Model artifacts are not automatically available in a fresh Git clone.
- The project is a fraud detection prototype, not a production banking or payment system.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
