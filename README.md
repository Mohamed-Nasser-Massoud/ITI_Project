# Hands-On Machine Learning Book Assistant

A local, cited RAG chatbot over the indexed *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* book. The notebook creates the persisted Chroma index; FastAPI owns retrieval and Ollama generation; Streamlit is the chat client.

## Architecture

```text
Streamlit chat UI (:8501)
        |
        | POST /query
        v
FastAPI backend (:8010)
   |                 |
   |                 +--> Ollama 0.34.0 (:11434)
   |                      llama3.2:3b
   |
   +--> SentenceTransformers embedding model
   |    all-MiniLM-L6-v2
   |
   +--> persisted Chroma collection
        backend/data/vector_store/
```

## Verified RAG artifact

The application uses the generated output copied from `/home/alihamdi/Downloads/results (3)`:

| Item | Value |
|---|---|
| Source | 1 PDF, 1,157 extracted page/records |
| Chunks | 2,687 |
| Chunk size | 900 characters |
| Chunk overlap | 150 characters |
| Vector database | Chroma, cosine distance |
| Embedding dimension | 384 |
| Collection | `hands_on_ml_book` |
| Retrieval | Top 4 chunks |
| Generation model | Ollama `llama3.2:3b` |

The source PDF is not redistributed by this project. The vector store is excluded from Git by default; the notebook configuration, provenance record, and evaluation CSV remain available under `backend/data/rag_output/`.

## Local layout

```text
.
├── backend/
│   ├── app/main.py
│   ├── app/api/routes/query.py
│   ├── app/core/config.py
│   ├── app/schemas/query.py
│   ├── app/services/retrieval.py
│   ├── app/services/generation.py
│   ├── data/vector_store/          # generated Chroma output
│   ├── data/rag_output/            # notebook config, provenance, and evaluation CSV
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── requirements.txt
├── notebooks/rag_pipeline.ipynb
├── scripts/
├── .tools/ollama/                  # local Ollama executable, not tracked
├── .ollama_models/                 # local Ollama model store, not tracked
└── .model_cache/                   # local embedding cache, not tracked
```

## Model and disk usage

The official Ollama registry lists `llama3.2:3b` at **2.0 GB**. The executable is already installed locally at `.tools/ollama/bin/ollama`; its model store is `.ollama_models`, not `~/.ollama`. The `all-MiniLM-L6-v2` embedding model is cached under `.model_cache` on first backend startup.

## Setup

The target directory is:

```text
/run/media/alihamdi/Projects/ITI/Graduation project ITI
```

The project already contains the local Ollama binary and Python virtual environment. If the binary is ever missing, the local installer downloads the official Linux archive into `.tools/ollama` without using `/usr/local`, `/home`, or `Downloads`:

```bash
cd "/run/media/alihamdi/Projects/ITI/Graduation project ITI"
./ollama-install.sh
```

Pull the generation model into the project-local store:

```bash
./scripts/pull_ollama_model.sh
```

Create the ignored local environment files:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

The backend dependencies are already installed in `.venv`. To reproduce them in that environment:

```bash
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/pip install -r frontend/requirements.txt
```

The first real backend startup may download `all-MiniLM-L6-v2` from Hugging Face. The launcher sets `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `.model_cache`, so this download stays on the project disk.

## Run

Open three terminals in the project directory.

### 1. Ollama

```bash
./scripts/serve_ollama.sh
```

### 2. FastAPI

```bash
./scripts/serve_backend.sh
```

The API is available at <http://127.0.0.1:8010/docs>.

### 3. Streamlit

```bash
./scripts/serve_frontend.sh
```

The chat UI is available at <http://127.0.0.1:8501>.

## API reference

### `GET /health`

Reports the API process, Chroma retrieval service, vector count, embedding model, Ollama reachability, and configured generation model separately.

### `GET /metadata`

Returns the notebook contract, model names, chunking settings, artifact paths, and limitations.

### `POST /query`

Request:

```bash
curl -s http://127.0.0.1:8010/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is overfitting?"}'
```

The response contains:

- `answer`: generated answer constrained to retrieved context
- `sources`: cited source-page/chunk labels
- `passages`: retrieved text, page, chunk ID, and Chroma distance for inspection

Blank questions and questions longer than 2,000 characters return `422`.

## Tests and verification

```bash
cd backend
../.venv/bin/pytest -q
cd ..
.venv/bin/python -m compileall -q backend frontend
```

The test suite covers the happy path, validation, component health, metadata, and retrieval evidence. A live verification should be performed with the three processes running:

```bash
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/metadata
curl -s http://127.0.0.1:8010/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why is feature scaling important for gradient descent?"}'
```

## Limitations

- Answers are only as good as the persisted notebook retrieval results and the local Ollama model.
- The notebook's original evaluation CSV contains generated answers with blank manual `relevant_context` and `grounded` columns; those fields still require human review before reporting evaluation accuracy.
- The application does not expose the copyrighted source PDF or rebuild the vector store during a user request.
- If Ollama is stopped or the configured model is missing, `/health` reports a degraded generation component and `/query` returns a readable service error rather than fabricating an answer.
