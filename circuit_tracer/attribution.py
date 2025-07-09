"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import contextlib
import logging
import time
import weakref
from functools import partial
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.disk_offload import offload_modules


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            activation_matrix, error_vectors, token_vectors, decoder_vecs, feature_output_hook
        )

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

    def _caching_hooks(self, feature_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            # Ensure activations require gradients for backward hooks
            if not acts.requires_grad:
                acts = acts.requires_grad_(True)
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> Tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            proxy._batch_buffer[write_index] = einsum(
                grads.to(output_vecs.dtype)[read_index],
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    def _make_attribution_hooks(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """Create the complete backward-hook for computing attribution scores."""

        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[start:end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return activation_matrix._nnz() + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        # Token-embedding nodes
        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[tok_start : tok_start + n_pos],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),
            bwd_hooks=self._attribution_hooks,
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
        debug: bool = False,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """
        
        if debug:
            print(f"BACKWARD PASS HEALTH CHECK: Starting compute_batch...")
            
            # Check input health
            inject_healthy = not (torch.isnan(inject_values).any() or torch.isinf(inject_values).any())
            print(f"BACKWARD PASS HEALTH CHECK: Inject values healthy: {inject_healthy}")
            if not inject_healthy:
                print(f"BACKWARD PASS HEALTH CHECK: ❌ INJECT VALUES CONTAIN NaN/Inf!")
                print(f"BACKWARD PASS HEALTH CHECK: Inject values range: [{inject_values.min():.6f}, {inject_values.max():.6f}]")
                print(f"BACKWARD PASS HEALTH CHECK: Inject NaN count: {torch.isnan(inject_values).sum()}")
                print(f"BACKWARD PASS HEALTH CHECK: Inject Inf count: {torch.isinf(inject_values).sum()}")
            
            # Check cached activations
            for i, act in enumerate(self._resid_activations):
                if act is not None:
                    act_healthy = not (torch.isnan(act).any() or torch.isinf(act).any())
                    if not act_healthy:
                        print(f"BACKWARD PASS HEALTH CHECK: ❌ CACHED ACTIVATION {i} CONTAINS NaN/Inf!")
                        print(f"BACKWARD PASS HEALTH CHECK: Act {i} range: [{act.min():.6f}, {act.max():.6f}]")
                        print(f"BACKWARD PASS HEALTH CHECK: Act {i} NaN count: {torch.isnan(act).sum()}")
                        break

        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[int(layer)].register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        result = buf.T[: len(layers)]
        
        if debug:
            print(f"BACKWARD PASS HEALTH CHECK: Backward pass completed, checking result...")
            result_healthy = not (torch.isnan(result).any() or torch.isinf(result).any())
            print(f"BACKWARD PASS HEALTH CHECK: Result healthy: {result_healthy}")
            if not result_healthy:
                print(f"BACKWARD PASS HEALTH CHECK: ❌ BACKWARD PASS RESULT CONTAINS NaN/Inf!")
                print(f"BACKWARD PASS HEALTH CHECK: Result range: [{result.min():.6f}, {result.max():.6f}]")
                print(f"BACKWARD PASS HEALTH CHECK: Result NaN count: {torch.isnan(result).sum()}")
                print(f"BACKWARD PASS HEALTH CHECK: Result Inf count: {torch.isinf(result).sum()}")
        
        return result


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


@torch.no_grad()
def select_scaled_decoder_vecs(
    activations: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    """

    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activations):
        _, feat_idx = row.coalesce().indices()
        rows.append(transcoders[layer].W_dec[feat_idx])
    return torch.cat(rows) * activations.values()[:, None]


@torch.no_grad()
def select_encoder_rows(
    activation_matrix: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""

    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        _, feat_idx = row.coalesce().indices()
        rows.append(transcoders[layer].W_enc.T[feat_idx])
    return torch.cat(rows)


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None, debug=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if debug:
        print(f"DEBUG: compute_partial_influences called with:")
        print(f"  edge_matrix shape: {edge_matrix.shape}")
        print(f"  logit_p shape: {logit_p.shape}")
        print(f"  row_to_node_index shape: {row_to_node_index.shape}")
        print(f"  max_iter: {max_iter}")
        
        # Analyze the raw edge matrix
        print(f"  Raw edge_matrix stats:")
        print(f"    Range: [{edge_matrix.min():.6f}, {edge_matrix.max():.6f}]")
        print(f"    Mean: {edge_matrix.mean():.6f}")
        print(f"    Std: {edge_matrix.std():.6f}")
        print(f"    NaN count: {torch.isnan(edge_matrix).sum()}")
        print(f"    Inf count: {torch.isinf(edge_matrix).sum()}")
        print(f"    Zero count: {(edge_matrix == 0).sum()}")
        print(f"    Sparsity: {(edge_matrix == 0).float().mean():.4f}")

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    if debug:
        print(f"  Normalized matrix stats:")
        print(f"    Range: [{normalized_matrix.min():.6f}, {normalized_matrix.max():.6f}]")
        print(f"    Mean: {normalized_matrix.mean():.6f}")
        print(f"    Std: {normalized_matrix.std():.6f}")
        print(f"    Row sums range: [{normalized_matrix.sum(dim=1).min():.6f}, {normalized_matrix.sum(dim=1).max():.6f}]")
        
        # Check if the matrix is triangular
        if normalized_matrix.shape[0] == normalized_matrix.shape[1]:
            upper_tri = torch.triu(normalized_matrix, diagonal=1)
            lower_tri = torch.tril(normalized_matrix, diagonal=-1)
            print(f"    Upper triangular norm: {upper_tri.norm():.6f}")
            print(f"    Lower triangular norm: {lower_tri.norm():.6f}")
            print(f"    Is approximately upper triangular: {lower_tri.norm() < 1e-6}")
        
        # Compute eigenvalues for small matrices
        if normalized_matrix.shape[0] <= 1000 and normalized_matrix.shape[0] == normalized_matrix.shape[1]:
            try:
                eigenvals = torch.linalg.eigvals(normalized_matrix.cpu())
                spectral_radius = eigenvals.abs().max().item()
                print(f"    Spectral radius: {spectral_radius:.6f}")
                print(f"    Largest eigenvalue (real): {eigenvals.real.max().item():.6f}")
                print(f"    Smallest eigenvalue (real): {eigenvals.real.min().item():.6f}")
                
                # Check if spectral radius is problematic
                if spectral_radius >= 1.0:
                    print(f"    ⚠️  PROBLEM: Spectral radius >= 1.0 will cause divergence!")
                    # Find the problematic eigenvalues
                    problem_eigs = eigenvals[eigenvals.abs() >= 1.0]
                    print(f"    Problematic eigenvalues: {problem_eigs}")
                    
            except Exception as e:
                print(f"    Could not compute eigenvalues: {e}")
        
        # Analyze row_to_node_index
        print(f"  row_to_node_index analysis:")
        print(f"    Range: [{row_to_node_index.min()}, {row_to_node_index.max()}]")
        print(f"    Is sorted: {torch.all(row_to_node_index[:-1] <= row_to_node_index[1:])}")
        print(f"    Unique values: {len(torch.unique(row_to_node_index))}")
        
        # Check for cycles introduced by row_to_node_index
        if len(row_to_node_index) <= 20:
            print(f"    row_to_node_index: {row_to_node_index.tolist()}")

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    if debug:
        print(f"  Initial prod stats:")
        print(f"    Range: [{prod.min():.6f}, {prod.max():.6f}]")
        print(f"    Norm: {prod.norm():.6f}")
        print(f"    Non-zero elements: {(prod != 0).sum()}")

    for iteration in range(max_iter):
        old_prod = prod.clone()
        prod = prod[row_to_node_index] @ normalized_matrix
        
        if debug and (iteration < 10 or iteration % 50 == 0):
            print(f"  Iteration {iteration}:")
            print(f"    prod norm: {prod.norm():.6f}")
            print(f"    prod range: [{prod.min():.6f}, {prod.max():.6f}]")
            print(f"    Change from previous: {(prod - old_prod).norm():.6f}")
            
            # Check for growth
            if iteration > 0:
                growth_ratio = prod.norm() / (old_prod.norm() + 1e-8)
                print(f"    Growth ratio: {growth_ratio:.6f}")
                if growth_ratio > 1.1:
                    print(f"    ⚠️  PROBLEM: Growth ratio > 1.1 indicates divergence!")
        
        if not prod.any():
            if debug:
                print(f"  Converged to zero after {iteration + 1} iterations")
            break
        influences += prod
    else:
        if debug:
            print(f"  FAILED to converge after {max_iter} iterations")
            print(f"  Final prod norm: {prod.norm():.6f}")
            print(f"  Final influences norm: {influences.norm():.6f}")
        raise RuntimeError("Failed to converge")

    if debug:
        print(f"  Final influences stats:")
        print(f"    Range: [{influences.min():.6f}, {influences.max():.6f}]")
        print(f"    Norm: {influences.norm():.6f}")
        print(f"    Non-zero elements: {(influences != 0).sum()}")

    return influences


def ensure_tokenized(prompt: Union[str, torch.Tensor, List[int]], tokenizer) -> torch.Tensor:
    """Convert *prompt* → 1-D tensor of token ids (no batch dim)."""

    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt").input_ids[0]
    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0) if prompt.ndim == 2 else prompt
    if isinstance(prompt, list):
        return torch.tensor(prompt, dtype=torch.long)
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    debug: bool = False,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.
        debug: Whether to enable detailed debugging output for convergence analysis.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
            debug=debug,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        logger.removeHandler(handler)


def _run_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    update_interval=4,
    logger=None,
    debug=False,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = ensure_tokenized(prompt, model.tokenizer)
    
    if debug:
        print(f"HEALTH CHECK: Input IDs: {input_ids.shape}")
        print(f"HEALTH CHECK: About to call model.setup_attribution...")
    
    logits, activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        input_ids, sparse=True
    )
    
    if debug:
        print(f"HEALTH CHECK: setup_attribution completed, checking outputs...")
        
        # Check logits
        logits_healthy = not (torch.isnan(logits).any() or torch.isinf(logits).any())
        print(f"HEALTH CHECK: Logits healthy: {logits_healthy}")
        if not logits_healthy:
            print(f"HEALTH CHECK: ❌ LOGITS CONTAIN NaN/Inf!")
            print(f"HEALTH CHECK: Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
            print(f"HEALTH CHECK: Logits NaN count: {torch.isnan(logits).sum()}")
            print(f"HEALTH CHECK: Logits Inf count: {torch.isinf(logits).sum()}")
        
        # Check activation matrix values
        act_values = activation_matrix.values()
        act_healthy = not (torch.isnan(act_values).any() or torch.isinf(act_values).any())
        print(f"HEALTH CHECK: Activation matrix values healthy: {act_healthy}")
        if not act_healthy:
            print(f"HEALTH CHECK: ❌ ACTIVATION MATRIX CONTAINS NaN/Inf!")
            print(f"HEALTH CHECK: Activation values range: [{act_values.min():.6f}, {act_values.max():.6f}]")
            print(f"HEALTH CHECK: Activation NaN count: {torch.isnan(act_values).sum()}")
            print(f"HEALTH CHECK: Activation Inf count: {torch.isinf(act_values).sum()}")
        
        # Check error vectors
        error_healthy = not (torch.isnan(error_vecs).any() or torch.isinf(error_vecs).any())
        print(f"HEALTH CHECK: Error vectors healthy: {error_healthy}")
        if not error_healthy:
            print(f"HEALTH CHECK: ❌ ERROR VECTORS CONTAIN NaN/Inf!")
            print(f"HEALTH CHECK: Error vectors range: [{error_vecs.min():.6f}, {error_vecs.max():.6f}]")
            print(f"HEALTH CHECK: Error vectors NaN count: {torch.isnan(error_vecs).sum()}")
            print(f"HEALTH CHECK: Error vectors Inf count: {torch.isinf(error_vecs).sum()}")
        
        # Check token vectors  
        token_healthy = not (torch.isnan(token_vecs).any() or torch.isinf(token_vecs).any())
        print(f"HEALTH CHECK: Token vectors healthy: {token_healthy}")
        if not token_healthy:
            print(f"HEALTH CHECK: ❌ TOKEN VECTORS CONTAIN NaN/Inf!")
            print(f"HEALTH CHECK: Token vectors range: [{token_vecs.min():.6f}, {token_vecs.max():.6f}]")
            print(f"HEALTH CHECK: Token vectors NaN count: {torch.isnan(token_vecs).sum()}")
            print(f"HEALTH CHECK: Token vectors Inf count: {torch.isinf(token_vecs).sum()}")
    
    decoder_vecs = select_scaled_decoder_vecs(activation_matrix, model.transcoders)
    encoder_rows = select_encoder_rows(activation_matrix, model.transcoders)
    
    if debug:
        print(f"HEALTH CHECK: Checking decoder and encoder vectors...")
        
        # Check decoder vectors
        decoder_healthy = not (torch.isnan(decoder_vecs).any() or torch.isinf(decoder_vecs).any())
        print(f"HEALTH CHECK: Decoder vectors healthy: {decoder_healthy}")
        if not decoder_healthy:
            print(f"HEALTH CHECK: ❌ DECODER VECTORS CONTAIN NaN/Inf!")
            print(f"HEALTH CHECK: Decoder vectors range: [{decoder_vecs.min():.6f}, {decoder_vecs.max():.6f}]")
            print(f"HEALTH CHECK: Decoder vectors NaN count: {torch.isnan(decoder_vecs).sum()}")
            print(f"HEALTH CHECK: Decoder vectors Inf count: {torch.isinf(decoder_vecs).sum()}")
        
        # Check encoder rows
        encoder_healthy = not (torch.isnan(encoder_rows).any() or torch.isinf(encoder_rows).any())
        print(f"HEALTH CHECK: Encoder rows healthy: {encoder_healthy}")
        if not encoder_healthy:
            print(f"HEALTH CHECK: ❌ ENCODER ROWS CONTAIN NaN/Inf!")
            print(f"HEALTH CHECK: Encoder rows range: [{encoder_rows.min():.6f}, {encoder_rows.max():.6f}]")
            print(f"HEALTH CHECK: Encoder rows NaN count: {torch.isnan(encoder_rows).sum()}")
            print(f"HEALTH CHECK: Encoder rows Inf count: {torch.isinf(encoder_rows).sum()}")
    
    ctx = AttributionContext(
        activation_matrix, error_vecs, token_vecs, decoder_vecs, model.feature_output_hook
    )
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        if debug:
            print(f"HEALTH CHECK: Starting layer-by-layer forward pass debugging...")
            expanded_input = input_ids.expand(batch_size, -1)
            print(f"HEALTH CHECK: Input shape: {expanded_input.shape}")
            
            # Check layer by layer to find where NaN is introduced
            current_residual = model.embed(expanded_input)
            
            embed_healthy = not (torch.isnan(current_residual).any() or torch.isinf(current_residual).any())
            print(f"HEALTH CHECK: After embedding healthy: {embed_healthy}")
            if not embed_healthy:
                print(f"HEALTH CHECK: ❌ EMBEDDING CONTAINS NaN/Inf!")
                print(f"HEALTH CHECK: Embedding range: [{current_residual.min():.6f}, {current_residual.max():.6f}]")
            
            # Check each transformer layer
            for layer_idx in range(model.cfg.n_layers):
                if embed_healthy or layer_idx <= 5:  # Only check first few layers if already corrupted
                    old_residual = current_residual.clone()
                    current_residual = model.blocks[layer_idx](current_residual)
                    
                    layer_healthy = not (torch.isnan(current_residual).any() or torch.isinf(current_residual).any())
                    print(f"HEALTH CHECK: After layer {layer_idx} healthy: {layer_healthy}")
                    
                    if not layer_healthy:
                        print(f"HEALTH CHECK: ❌ LAYER {layer_idx} INTRODUCED NaN/Inf!")
                        print(f"HEALTH CHECK: Layer {layer_idx} range: [{current_residual.min():.6f}, {current_residual.max():.6f}]")
                        print(f"HEALTH CHECK: Layer {layer_idx} NaN count: {torch.isnan(current_residual).sum()}")
                        
                        # Check if this is layer 3 (the LoRA adapted layer)
                        if layer_idx == 3:
                            print(f"HEALTH CHECK: 🎯 LAYER 3 IS THE LoRA ADAPTED LAYER!")
                            print(f"HEALTH CHECK: This confirms LoRA adapter is causing NaN values")
                            
                            # Check the difference introduced by this layer
                            layer_diff = current_residual - old_residual
                            diff_healthy = not (torch.isnan(layer_diff).any() or torch.isinf(layer_diff).any())
                            print(f"HEALTH CHECK: Layer 3 diff healthy: {diff_healthy}")
                            if not diff_healthy:
                                print(f"HEALTH CHECK: Layer 3 diff range: [{layer_diff.min():.6f}, {layer_diff.max():.6f}]")
                                print(f"HEALTH CHECK: Layer 3 diff NaN count: {torch.isnan(layer_diff).sum()}")
                        
                        break  # Stop checking once we find the problematic layer
            
            residual = current_residual
        else:
            residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        
        if debug:
            print(f"HEALTH CHECK: Checking residual before ln_final...")
            residual_healthy = not (torch.isnan(residual).any() or torch.isinf(residual).any())
            print(f"HEALTH CHECK: Residual before ln_final healthy: {residual_healthy}")
            if not residual_healthy:
                print(f"HEALTH CHECK: ❌ RESIDUAL BEFORE LN_FINAL CONTAINS NaN/Inf!")
                print(f"HEALTH CHECK: Residual range: [{residual.min():.6f}, {residual.max():.6f}]")
                print(f"HEALTH CHECK: Residual NaN count: {torch.isnan(residual).sum()}")
                print(f"HEALTH CHECK: Residual Inf count: {torch.isinf(residual).sum()}")
                
                # Check individual layers in residual
                if residual.ndim >= 2:
                    print(f"HEALTH CHECK: Checking residual by position...")
                    for pos in range(min(5, residual.shape[1])):  # Check first 5 positions
                        pos_residual = residual[:, pos, :]
                        pos_healthy = not (torch.isnan(pos_residual).any() or torch.isinf(pos_residual).any())
                        if not pos_healthy:
                            print(f"HEALTH CHECK: Position {pos} unhealthy - NaN: {torch.isnan(pos_residual).sum()}")
        
        final_residual = model.ln_final(residual)
        
        if debug:
            print(f"HEALTH CHECK: Checking residual after ln_final...")
            final_healthy = not (torch.isnan(final_residual).any() or torch.isinf(final_residual).any())
            print(f"HEALTH CHECK: Residual after ln_final healthy: {final_healthy}")
            if not final_healthy:
                print(f"HEALTH CHECK: ❌ RESIDUAL AFTER LN_FINAL CONTAINS NaN/Inf!")
                print(f"HEALTH CHECK: Final residual range: [{final_residual.min():.6f}, {final_residual.max():.6f}]")
                print(f"HEALTH CHECK: Final residual NaN count: {torch.isnan(final_residual).sum()}")
                print(f"HEALTH CHECK: Final residual Inf count: {torch.isinf(final_residual).sum()}")
                
                # Check ln_final parameters
                if hasattr(model.ln_final, 'weight') and hasattr(model.ln_final, 'bias'):
                    weight_healthy = not (torch.isnan(model.ln_final.weight).any() or torch.isinf(model.ln_final.weight).any())
                    bias_healthy = not (torch.isnan(model.ln_final.bias).any() or torch.isinf(model.ln_final.bias).any())
                    print(f"HEALTH CHECK: ln_final weight healthy: {weight_healthy}")
                    print(f"HEALTH CHECK: ln_final bias healthy: {bias_healthy}")
                    if not weight_healthy:
                        print(f"HEALTH CHECK: ln_final weight range: [{model.ln_final.weight.min():.6f}, {model.ln_final.weight.max():.6f}]")
                    if not bias_healthy:
                        print(f"HEALTH CHECK: ln_final bias range: [{model.ln_final.bias.min():.6f}, {model.ln_final.bias.max():.6f}]")
        
        # Ensure final_residual requires gradients for backward hooks
        if not final_residual.requires_grad:
            final_residual = final_residual.requires_grad_(True)
        
        ctx._resid_activations[-1] = final_residual
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    logger.info(
        f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
            debug=debug,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st], debug=debug
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=encoder_rows[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
                debug=debug,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph
