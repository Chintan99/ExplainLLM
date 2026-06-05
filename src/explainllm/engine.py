import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from explainllm.config import RelevanceConfig
from explainllm.utils import clean_token, clean_token_list, entropy
from explainllm.attention import attention_rollout


class LLMRelevanceWrapper(nn.Module):
    """
    Wrapper around a HuggingFace causal LM that computes per-token
    relevance scores during greedy generation.

    Relevance is computed over the FULL current context (prompt tokens +
    previously generated tokens) at each generation step.
    """

    def __init__(
        self,
        config: Optional[RelevanceConfig] = None,
        model: Optional[nn.Module] = None,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config or RelevanceConfig()

        if model is not None and tokenizer is not None:
            self.model = model.eval()
            self.tokenizer = tokenizer
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                dtype=self.config.get_torch_dtype(),
                token=self.config.auth_token or None,
                attn_implementation="eager",
            ).eval().to(self.config.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                token=self.config.auth_token or None,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = next(self.model.parameters()).device
        self._embed_hook_handle = None
        self._input_embeds = None

    # ------------------------------------------------------------------
    # Embedding hook helpers
    # ------------------------------------------------------------------

    def _get_embedding_layer(self) -> nn.Module:
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte
        raise AttributeError("Cannot auto-detect embedding layer.")

    def _register_embed_hook(self):
        def hook_fn(module, input, output):
            output.requires_grad_(True)
            output.retain_grad()
            self._input_embeds = output
        self._embed_hook_handle = self._get_embedding_layer().register_forward_hook(hook_fn)

    def _remove_embed_hook(self):
        if self._embed_hook_handle is not None:
            self._embed_hook_handle.remove()
            self._embed_hook_handle = None

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        return out.logits, out.attentions, out.hidden_states

    # ------------------------------------------------------------------
    # Generation with relevance
    # ------------------------------------------------------------------

    def generate_with_relevance(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        calculate_relevance: bool = True,
        relevance_method: Optional[str] = None,
        alpha: Optional[float] = None,
        verbose: bool = False,
        suppress_bos: Optional[bool] = None,
        bos_dampen: Optional[float] = None,
        use_top_layers_fraction: Optional[float] = None,
    ) -> Dict:
        """
        Generate tokens and compute per-step relevance over the FULL context.

        At step N the relevance vector has length (prompt_len + N), covering
        both original prompt tokens AND previously generated tokens.
        """
        cfg = self.config
        max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
        relevance_method = relevance_method or cfg.relevance_method
        alpha = alpha if alpha is not None else cfg.alpha
        suppress_bos = suppress_bos if suppress_bos is not None else cfg.suppress_bos
        bos_dampen = bos_dampen if bos_dampen is not None else cfg.bos_dampen
        use_top_layers_fraction = (
            use_top_layers_fraction if use_top_layers_fraction is not None
            else cfg.use_top_layers_fraction
        )

        tokenizer = self.tokenizer

        encoded = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        prompt_token_strings = clean_token_list(
            tokenizer.convert_ids_to_tokens(input_ids[0])
        )
        prompt_len = input_ids.shape[1]

        all_token_strings: List[str] = list(prompt_token_strings)

        # Detect BOS sink position(s)
        sink_positions: List[int] = []
        if suppress_bos and prompt_len > 0:
            first_id = input_ids[0, 0].item()
            bos_ids = set()
            if tokenizer.bos_token_id is not None:
                bos_ids.add(tokenizer.bos_token_id)
            if hasattr(tokenizer, "added_tokens_encoder"):
                for tok_str, tok_id in tokenizer.added_tokens_encoder.items():
                    if "begin" in tok_str.lower() or "bos" in tok_str.lower():
                        bos_ids.add(tok_id)
            if first_id in bos_ids:
                sink_positions.append(0)

        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()

        predicted_ids: List[int] = []
        relevance_scores: List[Dict] = []

        needs_grad = calculate_relevance and relevance_method in ("gradient", "combined")
        if needs_grad:
            self._register_embed_hook()

        try:
            for step in range(max_new_tokens):
                if needs_grad:
                    self.model.zero_grad(set_to_none=True)
                    if self._input_embeds is not None and self._input_embeds.grad is not None:
                        self._input_embeds.grad = None

                with torch.set_grad_enabled(needs_grad):
                    logits, attentions, hidden_states = self.forward(current_ids, current_mask)

                if attentions is None:
                    raise RuntimeError(
                        "attentions=None - load model with attn_implementation='eager'"
                    )
                if hidden_states is None:
                    raise RuntimeError("hidden_states=None - model doesn't support it")

                next_logits = logits[:, -1, :]
                next_id = torch.argmax(next_logits, dim=-1).item()
                predicted_ids.append(next_id)

                if verbose:
                    print(
                        f"  step {step:3d} | token {next_id:6d} | "
                        f"'{tokenizer.decode([next_id])}'"
                    )

                current_context_len = current_ids.shape[1]

                if calculate_relevance:
                    rel = self._compute_relevance(
                        logits=logits,
                        attentions=attentions,
                        hidden_states=hidden_states,
                        next_token_id=next_id,
                        context_len=current_context_len,
                        prompt_len=prompt_len,
                        method=relevance_method,
                        alpha=alpha,
                        sink_positions=sink_positions,
                        bos_dampen=bos_dampen,
                        use_top_layers_fraction=use_top_layers_fraction,
                    )
                    relevance_scores.append(rel)

                if next_id == tokenizer.eos_token_id:
                    break

                next_tensor = torch.tensor([[next_id]], device=self.device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)
                current_mask = torch.cat(
                    [current_mask, torch.ones_like(next_tensor)], dim=1
                )

                gen_tok_str = clean_token(
                    tokenizer.convert_ids_to_tokens([next_id])[0]
                )
                all_token_strings.append(gen_tok_str)

        finally:
            self._remove_embed_hook()

        generated_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        full_ids = input_ids[0].tolist() + predicted_ids
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)

        token_details = self._build_token_details(
            predicted_ids=predicted_ids,
            relevance_scores=relevance_scores,
            all_token_strings=all_token_strings,
            prompt_len=prompt_len,
            tokenizer=tokenizer,
        )

        return {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "generated_text": generated_text,
            "full_text": full_text,
            "prompt_tokens": list(prompt_token_strings),
            "prompt_len": prompt_len,
            "input_tokens": list(prompt_token_strings),
            "generated_tokens": predicted_ids,
            "all_token_strings": all_token_strings,
            "relevance_per_token": relevance_scores,
            "token_details": token_details,
        }

    # ------------------------------------------------------------------
    # Relevance computation
    # ------------------------------------------------------------------

    def _compute_relevance(
        self,
        logits: torch.Tensor,
        attentions: Tuple[torch.Tensor, ...],
        hidden_states: Tuple[torch.Tensor, ...],
        next_token_id: int,
        context_len: int,
        prompt_len: int,
        method: str = "combined",
        alpha: float = 0.5,
        sink_positions: Optional[List[int]] = None,
        bos_dampen: float = 0.01,
        use_top_layers_fraction: float = 0.5,
    ) -> Dict:
        results: Dict = {}
        results["context_len"] = context_len
        results["prompt_len"] = prompt_len

        if method in ("attention", "combined"):
            results["attention_relevance"] = self._attention_relevance(
                attentions, context_len,
                sink_positions=sink_positions,
                bos_dampen=bos_dampen,
                use_top_layers_fraction=use_top_layers_fraction,
            )

        if method in ("gradient", "combined"):
            results["gradient_relevance"] = self._gradient_relevance(
                logits, next_token_id, context_len,
            )

        if method == "rollout":
            results["rollout_relevance"] = self._rollout_relevance(
                attentions, context_len,
                sink_positions=sink_positions,
                sink_dampen=bos_dampen,
            )

        if method == "attention":
            final = results["attention_relevance"]
        elif method == "gradient":
            final = results["gradient_relevance"]
        elif method == "rollout":
            final = results["rollout_relevance"]
        elif method == "combined":
            final = (
                alpha * results["attention_relevance"]
                + (1 - alpha) * results["gradient_relevance"]
            )
            final = final / (final.sum() + 1e-10)
        else:
            raise ValueError(f"Unknown method: {method}")

        results["input_explainllm"] = final
        results["prompt_relevance"] = final[:prompt_len]
        results["generated_relevance"] = final[prompt_len:]
        results["layer_relevance"] = self._layer_relevance(attentions, context_len)
        return results

    def _attention_relevance(
        self,
        attentions: Tuple[torch.Tensor, ...],
        context_len: int,
        sink_positions: Optional[List[int]] = None,
        bos_dampen: float = 0.01,
        use_top_layers_fraction: float = 0.5,
    ) -> torch.Tensor:
        num_layers = len(attentions)
        if num_layers == 0:
            raise ValueError("No attention tensors were provided.")

        # Ensure at least one layer is selected even for edge-case fractions.
        start_layer = max(0, int(num_layers * (1.0 - use_top_layers_fraction)))
        selected = attentions[start_layer:] or attentions[-1:]

        stacked = torch.stack(selected, dim=0)
        last_tok = stacked[:, 0, :, -1, :context_len]

        head_ent = entropy(last_tok, dim=-1)
        max_ent = math.log(context_len) if context_len > 1 else 1.0
        weights = (1.0 - head_ent / (max_ent + 1e-10)).clamp(min=0.0)
        weight_sum = weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum <= 1e-10:
            # Degenerate case: all selected heads are near-max entropy.
            # Fall back to uniform head/layer weighting.
            weights = torch.ones_like(weights) / weights.numel()
        else:
            weights = weights / weight_sum

        mean = (last_tok * weights.unsqueeze(-1)).sum(dim=(0, 1))

        if sink_positions:
            for pos in sink_positions:
                if pos < context_len:
                    mean[pos] *= bos_dampen

        mean_sum = mean.sum()
        if not torch.isfinite(mean_sum) or mean_sum <= 1e-10:
            mean = torch.ones_like(mean) / context_len
        else:
            mean = mean / mean_sum
        return mean.detach().cpu()

    def _gradient_relevance(
        self,
        logits: torch.Tensor,
        next_token_id: int,
        context_len: int,
    ) -> torch.Tensor:
        embeds = self._input_embeds

        if embeds is not None and embeds.requires_grad:
            target = logits[0, -1, next_token_id]
            if embeds.grad is not None:
                embeds.grad.zero_()
            target.backward(retain_graph=True)

            if embeds.grad is not None:
                grad = embeds.grad[0, :context_len, :]
                act = embeds[0, :context_len, :].detach()
                relevance = (grad * act).norm(dim=-1)
                relevance = relevance / (relevance.sum() + 1e-10)
                return relevance.detach().cpu()

        return torch.ones(context_len) / context_len

    def _rollout_relevance(
        self,
        attentions: Tuple[torch.Tensor, ...],
        context_len: int,
        sink_positions: Optional[List[int]] = None,
        sink_dampen: float = 0.1,
    ) -> torch.Tensor:
        rollout_matrix = attention_rollout(attentions, sink_positions, sink_dampen)
        rel = rollout_matrix[0, -1, :context_len]
        return (rel / (rel.sum() + 1e-10)).detach().cpu()

    def _layer_relevance(
        self,
        attentions: Tuple[torch.Tensor, ...],
        context_len: int,
    ) -> List[torch.Tensor]:
        layers = []
        for att in attentions:
            rel = att[0, :, -1, :context_len].mean(dim=0)
            rel = rel / (rel.sum() + 1e-10)
            layers.append(rel.detach().cpu())
        return layers

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_token_details(
        predicted_ids: List[int],
        relevance_scores: List[Dict],
        all_token_strings: List[str],
        prompt_len: int,
        tokenizer,
        top_k: int = 5,
    ) -> List[Dict]:
        details = []
        for i, tid in enumerate(predicted_ids):
            entry: Dict = {
                "step": i,
                "generated_token": tokenizer.decode([tid]),
                "token_id": tid,
            }
            if i < len(relevance_scores):
                rel = relevance_scores[i]["input_explainllm"]
                ctx_len = len(rel)
                context_tokens = all_token_strings[:ctx_len]
                k = min(top_k, ctx_len)
                top_indices = torch.topk(rel, k).indices.tolist()

                entry["top_contributing_tokens"] = [
                    {
                        "token": context_tokens[idx],
                        "position": idx,
                        "is_prompt": idx < prompt_len,
                        "relevance": rel[idx].item(),
                    }
                    for idx in top_indices
                ]
                entry["top_contributing_input_tokens"] = entry["top_contributing_tokens"]
                entry["full_relevance"] = rel.tolist()
                entry["context_tokens"] = context_tokens
                entry["prompt_len"] = prompt_len

            details.append(entry)
        return details
