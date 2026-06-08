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

## Technical Working

ExplainLLM works on decoder-only causal language models loaded through Hugging Face `AutoModelForCausalLM`. It explains autoregressive, greedy generation step by step: for every generated token, it computes a normalized relevance score over the full context available at that step, including both the original prompt tokens and any tokens generated earlier.

### How Explainability Is Generated

For a prompt such as `What is AI?`, the wrapper first tokenizes the prompt and runs the model with:

- `output_attentions=True`
- `output_hidden_states=True`
- `attn_implementation="eager"`

At generation step `N`, the current context is:

```text
prompt tokens + previously generated tokens
```

The next token is selected greedily with `argmax` over the final-position logits. ExplainLLM then assigns relevance scores to every token in the current context. Each relevance vector is normalized so the scores sum to approximately `1.0`.

The implementation supports four relevance methods:

- `attention`: uses attention from the last token position over the top transformer layers. Heads with lower entropy, meaning more focused attention, receive more weight. Beginning-of-sequence attention sinks can be dampened with `suppress_bos` and `bos_dampen`.
- `gradient`: uses gradient x activation saliency. The score for the selected next-token logit is backpropagated to the input embeddings, then `grad * activation` is reduced per token.
- `rollout`: multiplies attention flow across layers with residual identity mixing to estimate how information propagates through the transformer stack.
- `combined`: blends attention relevance and gradient relevance as:

```text
final = alpha * attention_relevance + (1 - alpha) * gradient_relevance
```

The default method is `combined` with `alpha=0.5`.

### Supported Models

ExplainLLM supports Hugging Face models that satisfy these requirements:

- They can be loaded with `AutoModelForCausalLM.from_pretrained(...)`.
- They are decoder-only causal language models, not masked language models, sequence classifiers, or encoder-decoder models.
- They return attention tensors when called with `output_attentions=True`.
- Their embedding layer can be auto-detected as either:
  - `model.model.embed_tokens`, used by LLaMA-style architectures.
  - `model.transformer.wte`, used by GPT-2-style architectures.

This means the package is intended for transformer causal-LM families such as:

| Architecture family | Example model IDs |
|---------------------|-------------------|
| LLaMA-style | `meta-llama/Llama-3.2-1B-Instruct` |
| Mistral-style | `mistralai/Mistral-7B-Instruct-v0.3` |
| Gemma-style | `google/gemma-2-2b-it` |
| Qwen-style | `Qwen/Qwen2.5-0.5B-Instruct` |
| GPT-2-style | `gpt2`, `distilgpt2`, `sshleifer/tiny-gpt2` |

Large gated models may require `HF_TOKEN`. GPU execution is recommended for larger models; CPU is practical mainly for small models such as `sshleifer/tiny-gpt2`, `distilgpt2`, or similarly small causal LMs.

Models are not currently supported if they require a different embedding path, do not expose attentions, are encoder-decoder models such as T5 or BART, are BERT-style masked language models, are classifier-only models, or are multimodal models.

### Output Format

`calculate_relevance(...)` and `LLMRelevanceWrapper.generate_with_relevance(...)` return a dictionary with fields such as:

| Field | Meaning |
|-------|---------|
| `prompt` | Original input prompt |
| `generated_text` | Text generated by the model |
| `full_text` | Prompt plus generated text |
| `prompt_tokens` | Tokenized prompt after token cleanup |
| `generated_tokens` | Generated token IDs |
| `all_token_strings` | Prompt tokens plus generated token strings |
| `relevance_per_token` | Raw per-step relevance objects, including attention, gradient, rollout, combined scores, and layer-level attention relevance where applicable |
| `token_details` | Human-friendly per-step explanation data |

Each item in `token_details` corresponds to one generated token:

```json
{
  "step": 0,
  "generated_token": " Paris",
  "token_id": 1234,
  "top_contributing_tokens": [
    {
      "token": "France",
      "position": 5,
      "is_prompt": true,
      "relevance": 0.42
    }
  ],
  "full_relevance": [0.02, 0.03, 0.06, 0.21, 0.03, 0.42, 0.23],
  "context_tokens": ["What", " is", " the", " capital", " of", " France", "?"],
  "prompt_len": 7
}
```

The exact tokens and numbers depend on the model tokenizer and generated output.

### How To Interpret The Output

A higher relevance score means that, according to the selected relevance method, that context token contributed more strongly to the model's next-token decision at that generation step. Scores are relative within one step, so compare tokens inside the same `full_relevance` vector rather than comparing raw scores across unrelated prompts.

In text visualizations, tokens are marked as:

- `P`: original prompt token
- `G`: token generated in an earlier step

For example:

```text
Step 0 | Generated: ' Paris'
1. [P:5] France    0.42
2. [P:3] capital   0.21
3. [P:1] is        0.08
```

This means the token `France` had the strongest relevance for generating `Paris` at step `0`, followed by `capital`. In later steps, generated tokens can also become contributors, which lets you inspect whether the model is relying on the original prompt or on its own previous output.

The visualization utilities present the same data in different forms:

- Text bars show top contributing tokens per generated token.
- Heatmaps show generated tokens as rows and context tokens as columns.
- HTML highlights use stronger color intensity for higher relevance.
- Graph visualizations show high-relevance links between generated tokens and their contributing context tokens.

These scores are explanation signals, not formal causal proof. They are best used to debug model behavior, compare prompts, inspect whether important prompt tokens were used, and identify cases where generation depends heavily on earlier generated tokens or special-token attention sinks.

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
