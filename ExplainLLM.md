# ExplainLLM: How It Explains LLM Outputs

## Goal
ExplainLLM attributes each generated token to the tokens currently in context (prompt + previously generated tokens), step by step during autoregressive decoding.

It answers:
- Which earlier tokens influenced this next token the most?
- Is the model relying on prompt tokens, generated tokens, or both?

## Core Idea
For each generation step `t`, ExplainLLM computes a relevance vector over the current context:

- Context at step `t`: `x_1 ... x_C`
- Next token chosen by model: `y_t`
- Output relevance: `r_t in R^C`, where `sum(r_t) = 1`

Each `r_t[i]` is the estimated contribution of context token `x_i` to `y_t`.

## End-to-End Algorithm
1. Tokenize prompt and start greedy generation loop.
2. At each step:
   - Run forward pass with `output_attentions=True` and `output_hidden_states=True`.
   - Select next token via `argmax(logits[:, -1, :])`.
   - Compute per-token relevance over the current context using one method:
     - `attention`
     - `gradient`
     - `rollout`
     - `combined`
3. Store:
   - full relevance vector,
   - top contributing tokens,
   - context tokens at that step.
4. Append generated token to context and continue.

This produces a full stepwise trace in `token_details`.

## Relevance Methods

### 1) Attention Relevance
Uses last-token attention from selected upper transformer layers.

Steps:
1. Select top layers by `use_top_layers_fraction`.
2. Extract attention from the last query position to all context positions.
3. Compute entropy per head and weight heads by confidence:
   - `w = max(0, 1 - H(attn)/log(C))`
4. Apply sink dampening (for BOS-like sink tokens).
5. Normalize to sum to 1.

Why: low-entropy heads are often more selective and informative.

### 2) Gradient Relevance
Uses gradient x activation at the input embedding layer.

Steps:
1. Backprop from selected next-token logit.
2. For each context token embedding `e_i` with gradient `g_i`:
   - `score_i = || g_i * e_i ||_2`
3. Normalize scores to sum to 1.

Why: estimates local sensitivity of next-token score to each input token representation.

### 3) Attention Rollout Relevance
Computes effective multi-layer attention flow.

Steps:
1. Average heads per layer.
2. Dampen sink columns.
3. Add residual mix (`0.5 * A + 0.5 * I`) and renormalize.
4. Multiply across layers (rollout).
5. Take last-token row and normalize.

Why: approximates how information can flow across layers, not just a single layer snapshot.

### 4) Combined Relevance
Blends attention and gradient signals:

`r = alpha * r_attention + (1 - alpha) * r_gradient`

Then renormalizes.

Why: attention gives structural routing; gradient gives local causal sensitivity signal.

## What Gets Exported
Per generation step, ExplainLLM stores:
- generated token and id,
- full context token list at that step,
- full relevance vector (`full_relevance`),
- top contributors with positions and prompt/generated tags.

This makes explanations auditable and easy to visualize in text, HTML, heatmaps, and graphs.

## Why This Is Useful
1. Prompt debugging: see which prompt fragments drive each output token.
2. Drift detection: identify when generation starts self-conditioning too heavily on earlier generated tokens.
3. Safety and policy checks: inspect whether risky output traces to specific context fragments.
4. Model behavior analysis: compare methods (`attention`, `gradient`, `rollout`, `combined`) across tasks.
5. Communication: turn token-level internals into visual artifacts for stakeholders.

## Practical Caveats
1. Attribution is approximate, not a formal proof of causality.
2. Attention is not equivalent to explanation; it is one signal.
3. Gradient methods are local and can be noisy.
4. Results depend on decoding strategy, model family, and tokenization.

## Future Work: Stronger Explanation Methods
The following methods are good candidates to add next:

1. Integrated Gradients
   - More stable than plain grad x activation.
   - Add baseline path integration for each generation step.

2. Leave-One-Out / Token Occlusion
   - Remove or mask one token and measure next-token logit change.
   - Strong perturbation-based importance signal.

3. SHAP-style Approximations
   - Model-agnostic contribution estimates for token subsets.
   - Useful for cross-method validation (with sampling approximations).

4. LRP / DeepLIFT for Transformers
   - Layer-wise relevance backprop rules.
   - Can provide complementary explanations to gradients.

5. Activation Patching / Causal Mediation
   - Replace activations and measure output deltas.
   - Stronger mechanism-level causal evidence.

6. Logit Lens / Tuned Lens
   - Inspect intermediate layer predictions over the vocabulary.
   - Helps explain when and where decisions emerge.

7. Attention Attribution Variants
   - Grad-Attention, attention flow with gradient weighting, and path attributions.
   - Better than raw attention-only in many settings.

## Suggested Roadmap
1. Add Integrated Gradients first (best effort vs complexity tradeoff).
2. Add token occlusion for perturbation-based validation.
3. Build a method-comparison report per step (agreement/disagreement diagnostics).
4. Add causal activation patching for high-value deep-dive analyses.

