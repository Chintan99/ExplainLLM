"""Streamlit chat UI for ExplainLLM with sidebar prompt inspection."""

import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import streamlit as st

st.set_page_config(page_title="ExplainLLM", layout="wide")


@st.cache_resource
def load_model(model_id: str, device: str, dtype: str, token: str):
    """Cache model instances across reruns."""
    from explainllm.config import RelevanceConfig
    from explainllm.engine import LLMRelevanceWrapper

    config = RelevanceConfig(
        model_id=model_id,
        device=device,
        torch_dtype=dtype,
        auth_token=token,
    )
    return LLMRelevanceWrapper(config=config)


def _init_state() -> None:
    if "runs" not in st.session_state:
        st.session_state.runs = []
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None


def _create_plot(plot_fn, **kwargs) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    plot_fn(save_path=path, **kwargs)
    return path


def _post_backend(backend_url: str, endpoint: str, payload: Dict) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{backend_url}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(
            f"Backend HTTP {exc.code} on {endpoint}: {detail}"
        ) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Backend connection failed: {exc}") from exc


def _get_backend(backend_url: str, endpoint: str) -> Dict:
    req = urlrequest.Request(
        f"{backend_url}{endpoint}",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(
            f"Backend HTTP {exc.code} on {endpoint}: {detail}"
        ) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Backend connection failed: {exc}") from exc


def _find_run(run_id: Optional[str]) -> Optional[Dict]:
    if run_id is None:
        return None
    for run in st.session_state.runs:
        if run["id"] == run_id:
            return run
    return None


def _latest_run() -> Optional[Dict]:
    if not st.session_state.runs:
        return None
    return st.session_state.runs[-1]


def _render_sidebar_inspector(run: Optional[Dict]) -> None:
    if run is None:
        st.sidebar.info("Run a prompt, then select it to inspect relevance.")
        return

    from explainllm.export import export_relevance_json
    from explainllm.viz.graph import (
        plot_force_graph,
        plot_relevance_graph,
        plot_step_graph,
    )
    from explainllm.viz.heatmap import (
        plot_layer_relevance_heatmap,
        plot_relevance_heatmap,
        plot_single_step_heatmap,
    )
    from explainllm.viz.html import relevance_to_html
    from explainllm.viz.text import visualize_all_steps

    result = run["result"]
    n_steps = len(result.get("token_details", []))

    st.sidebar.subheader("Selected Prompt")
    st.sidebar.caption(f"Run {run['created_at']} | model={run['config']['model_id']}")
    st.sidebar.markdown(run["prompt"])

    if n_steps == 0:
        st.sidebar.warning("No generation steps available for visualization.")
        return

    step_idx = st.sidebar.slider(
        "Inspect step",
        min_value=0,
        max_value=n_steps - 1,
        value=0,
        key=f"inspect_step_{run['id']}",
    )

    with st.sidebar.expander("Text Explanation", expanded=False):
        st.code(visualize_all_steps(result, top_k=5), language=None)

    with st.sidebar.expander("Heatmaps", expanded=True):
        full_heatmap = _create_plot(
            plot_relevance_heatmap,
            result=result,
            title="Input -> Output Token Relevance",
        )
        st.image(full_heatmap, use_container_width=True)

        step_bar = _create_plot(
            plot_single_step_heatmap,
            result=result,
            step=step_idx,
        )
        st.image(step_bar, use_container_width=True)

        layer_heatmap = _create_plot(
            plot_layer_relevance_heatmap,
            result=result,
            step=step_idx,
        )
        st.image(layer_heatmap, use_container_width=True)

    with st.sidebar.expander("Graphs", expanded=False):
        arc_graph = _create_plot(plot_relevance_graph, result=result, top_k=3)
        st.image(arc_graph, use_container_width=True)

        radial_graph = _create_plot(
            plot_step_graph,
            result=result,
            step=step_idx,
            top_k=6,
        )
        st.image(radial_graph, use_container_width=True)

        try:
            force_graph = _create_plot(plot_force_graph, result=result, top_k=3)
            st.image(force_graph, use_container_width=True)
        except Exception as exc:
            st.info(f"Force graph unavailable: {exc}")

    with st.sidebar.expander("HTML Explanation", expanded=False):
        html = relevance_to_html(result, step=step_idx)
        st.markdown(html, unsafe_allow_html=True)

    with st.sidebar.expander("JSON Export", expanded=False):
        json_str = export_relevance_json(result)
        st.download_button(
            "Download JSON",
            data=json_str,
            file_name=f"relevance_{run['id']}.json",
            mime="application/json",
            key=f"download_{run['id']}",
            use_container_width=True,
        )
        st.json(json.loads(json_str))


def main():
    _init_state()
    backend_url = os.environ.get("BACKEND_URL", "").strip().rstrip("/")

    st.title("ExplainLLM Chat")
    st.caption("Send prompts and inspect each run from the sidebar.")

    with st.sidebar:
        st.header("Settings")
        default_model_id = os.environ.get(
            "MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"
        )
        model_options = [default_model_id]
        if backend_url:
            try:
                models_resp = _get_backend(backend_url, "/models")
                default_model_id = models_resp.get("default_model_id", default_model_id)
                loaded = models_resp.get("loaded_models", [])
                loaded_ids = [m.get("model_id", "") for m in loaded if m.get("model_id")]
                model_options = list(dict.fromkeys([default_model_id] + loaded_ids))
            except Exception as exc:
                st.warning(f"Could not fetch models from backend: {exc}")

        selected_model = st.selectbox(
            "Available models",
            options=model_options + ["Custom HuggingFace ID"],
            index=0,
        )
        if selected_model == "Custom HuggingFace ID":
            model_id = st.text_input(
                "Model ID",
                value=default_model_id,
                help="HuggingFace model identifier",
            ).strip()
        else:
            model_id = selected_model
        hf_token = st.text_input(
            "HF Token",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
        )
        device = st.selectbox("Device", ["cpu", "cuda", "cuda:0", "cuda:1"], index=0)
        dtype = st.selectbox("Dtype", ["float32", "float16", "bfloat16"], index=0)
        max_new_tokens = st.slider("Max new tokens", 1, 256, 20)
        method = st.selectbox(
            "Relevance method",
            ["combined", "attention", "gradient", "rollout"],
            index=0,
        )
        alpha = st.slider("Alpha (combined mode)", 0.0, 1.0, 0.5, step=0.05)
        if backend_url:
            st.caption(f"Backend: {backend_url}")
        else:
            st.caption("Backend: disabled (local inference in Streamlit process)")

        if st.button("Load Model", use_container_width=True):
            try:
                if backend_url:
                    _post_backend(
                        backend_url,
                        "/warmup",
                        {
                            "model_id": model_id,
                            "device": device,
                            "dtype": dtype,
                            "hf_token": hf_token,
                        },
                    )
                else:
                    load_model(model_id, device, dtype, hf_token)
                st.success("Model loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")

        if st.button("Clear History", use_container_width=True):
            st.session_state.runs = []
            st.session_state.selected_run_id = None
            st.rerun()

        st.divider()
        st.subheader("Prompts")
        if not st.session_state.runs:
            st.caption("No prompts yet.")
        else:
            for run in reversed(st.session_state.runs):
                label = run["prompt"].strip().replace("\n", " ")
                if len(label) > 56:
                    label = f"{label[:53]}..."
                is_selected = run["id"] == st.session_state.selected_run_id
                if st.button(
                    label,
                    key=f"pick_{run['id']}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_run_id = run["id"]
                    st.rerun()

    default_prompt = os.environ.get("PROMPT", "What is the capital of France?")
    prompt = st.chat_input(default_prompt)

    if prompt is not None and prompt.strip():
        if not hf_token and "llama" in model_id.lower():
            st.warning(
                "Gated models like LLaMA need HF_TOKEN in sidebar or environment."
            )

        with st.spinner(f"Running model: {model_id}"):
            try:
                if backend_url:
                    result = _post_backend(
                        backend_url,
                        "/relevance",
                        {
                            "prompt": prompt.strip(),
                            "model_id": model_id,
                            "device": device,
                            "dtype": dtype,
                            "max_new_tokens": max_new_tokens,
                            "method": method,
                            "alpha": alpha,
                            "hf_token": hf_token,
                        },
                    )
                else:
                    wrapper = load_model(model_id, device, dtype, hf_token)
                    result = wrapper.generate_with_relevance(
                        prompt=prompt.strip(),
                        max_new_tokens=max_new_tokens,
                        calculate_relevance=True,
                        relevance_method=method,
                        alpha=alpha,
                        verbose=False,
                    )
            except Exception as exc:
                st.error(f"Error during generation: {exc}")
                result = None

        if result is not None:
            run_id = uuid.uuid4().hex[:8]
            run_entry = {
                "id": run_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt.strip(),
                "result": result,
                "config": {
                    "model_id": model_id,
                    "device": device,
                    "dtype": dtype,
                    "method": method,
                    "alpha": alpha,
                    "max_new_tokens": max_new_tokens,
                },
            }
            st.session_state.runs.append(run_entry)
            st.session_state.selected_run_id = run_id
            st.rerun()

    if not st.session_state.runs:
        st.info("No runs yet. Enter a prompt in the chat input to start.")
    else:
        for run in st.session_state.runs:
            with st.chat_message("user"):
                st.markdown(run["prompt"])
            with st.chat_message("assistant"):
                st.markdown(run["result"].get("generated_text", ""))
                st.caption(
                    f"model={run['config']['model_id']} | "
                    f"method={run['config']['method']} | "
                    f"max_new_tokens={run['config']['max_new_tokens']}"
                )
                if st.button("Inspect in sidebar", key=f"inspect_{run['id']}"):
                    st.session_state.selected_run_id = run["id"]
                    st.rerun()

    selected_run = _find_run(st.session_state.selected_run_id)
    if selected_run is None:
        selected_run = _latest_run()
        if selected_run is not None:
            st.session_state.selected_run_id = selected_run["id"]

    _render_sidebar_inspector(selected_run)


if __name__ == "__main__":
    main()
