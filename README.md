# ExplainLLM

LLM Token Relevance Calculator using DL-Backtrace patterns. Computes and visualizes how each input token contributes to each generated token during autoregressive generation.

## Features

- **Multiple relevance methods**: attention, gradient, rollout, and combined
- **Full context tracking**: relevance is computed over the growing context (prompt + generated tokens)
- **Rich visualizations**: heatmaps, arc diagrams, radial graphs, force-directed graphs, HTML highlights
- **CLI and library usage**: use as a Python package or from the command line
- **Streamlit web UI**: interactive browser-based interface
- **Docker support**: containerized deployment with GPU passthrough

## Installation

```bash
# Core (torch + transformers + numpy)
pip install -e .

# With visualization support (matplotlib, seaborn, networkx)
pip install -e ".[viz]"

# With Streamlit web UI
pip install -e ".[app]"

# Everything
pip install -e ".[all]"
```

## Setup

Set your HuggingFace token (required for gated models like LLaMA):

```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

### Python API

```python
from explainllm import calculate_relevance, RelevanceConfig

config = RelevanceConfig(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    device="cuda",
    torch_dtype="float16",
    max_new_tokens=20,
)

result = calculate_relevance("What is the capital of France?", config=config)
print(result["generated_text"])

# Visualize
from explainllm.viz import visualize_all_steps, plot_relevance_heatmap

print(visualize_all_steps(result, top_k=5))
plot_relevance_heatmap(result, save_path="heatmap.png")
```

### CLI

```bash
# Basic usage
explainllm run --prompt "What is AI?" --visualize

# With options
explainllm run \
  --prompt "What is the capital of France?" \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device cuda \
  --dtype float16 \
  --method combined \
  --alpha 0.6 \
  --export results.json \
  --heatmap heatmap.png \
  --step0-bar relevance_step0_bar.png \
  --verbose
```

### Streamlit Web UI

```bash
streamlit run src/streamlit_app/app.py
```

The Streamlit app uses a chat-style interface:
- Send prompts from the chat input.
- Select or type a HuggingFace model ID in sidebar settings.
- Click any saved prompt in the sidebar to inspect that run.
- Sidebar inspector shows full relevance visuals (heatmaps, graphs, HTML, text, JSON).
- The layout is Open WebUI-inspired: chat in main panel, run-level inspection in side panel.

### Docker

```bash
# Dockerfiles:
# - Dockerfile.ui
# - Dockerfile.backend
# - Dockerfile.modelservice
#
# Create .env file with your token and defaults
echo "HF_TOKEN=hf_your_token_here" > .env
echo "MODEL_ID=meta-llama/Llama-3.2-1B-Instruct" >> .env
echo "PROMPT=What is the capital of France?" >> .env
echo "BACKEND_URL=http://backend:8000" >> .env
echo "MODEL_SERVICE_URL=http://modelservice:8001" >> .env

# Run all 3 services:
# 1) ui (Streamlit)
# 2) backend (gateway/orchestrator)
# 3) modelservice (ExplainLLM inference + model cache)
docker compose up

# Access:
# - UI:           http://localhost:8501
# - Backend API:  http://localhost:8000/health
# - Model API:    http://localhost:8001/health

# Run CLI (profile)
docker compose --profile cli run cli run --prompt "What is AI?" --visualize

# Override model/prompt at runtime
MODEL_ID=sshleifer/tiny-gpt2 PROMPT="Explain transformers" docker compose up
```

## Relevance Methods

| Method | Description |
|--------|-------------|
| `attention` | Entropy-weighted attention from top transformer layers with BOS-sink dampening |
| `gradient` | Gradient x activation (input saliency) |
| `rollout` | Attention rollout across all layers |
| `combined` | Weighted blend of attention and gradient (controlled by `alpha`) |

## Project Structure

```
src/explainllm/
  __init__.py          # Public API
  config.py            # RelevanceConfig dataclass
  utils.py             # Token cleaning, to_numpy, entropy
  attention.py         # Attention rollout
  engine.py            # LLMRelevanceWrapper (core)
  model_service.py     # Model service (cached ExplainLLM inference)
  backend.py           # Backend gateway service
  api.py               # Backward-compatible alias to model_service app
  convenience.py       # calculate_relevance() one-liner
  export.py            # JSON export
  cli.py               # CLI entry point
  viz/
    text.py            # Text bar charts
    html.py            # HTML inline visualization
    heatmap.py         # Matplotlib heatmaps
    graph.py           # Arc, radial, force-directed graphs
    stepwise.py        # Step-wise data extraction
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
