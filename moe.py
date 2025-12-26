import torch
import torch.nn as nn
import torch.nn.functional as F

"""
use_moe: bool = False,
num_experts_per_tok: int = 2,
n_routed_experts: int = 4,
n_shared_experts: int = 1,
scoring_func: str = 'softmax',
aux_loss_alpha: float = 0.01,
seq_aux: bool = True,
norm_topk_prob: bool = True,
"""

class MOEFeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])

        self.gate = MOEGate(config)

        if config.shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape

        bsz, seq_len, h = x.shape

        topk_idx, topk_weight, aux_loss = self.gate(x)
        # Shapes:
        #   topk_idx:      (bsz * seq_len, top_k)         # Index of top-k experts per token
        #   topk_weight:   (bsz * seq_len, top_k)         # Corresponding probabilities/weights per expert per token
        #   aux_loss:      Scalar (single value, usually float) for regularization

        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        # This block implements the token routing and computation for Mixture-of-Experts (MoE) during training.
        # Here's the step-by-step explanation:
        #
        # 1. `x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)`:
        #      For each input token, we need to pass it to multiple experts (as determined by the gating network).
        #      `repeat_interleave` effectively repeats each token's embedding as many times as there are experts assigned to each token.
        #      So if batch*seq = 3 and num_experts_per_tok = 2, then a 3xH tensor becomes a 6xH tensor, where each token appears twice in a row.
        #
        # 2. Initialize an empty output tensor `y` the same shape and type as `x`.
        #      This will be filled in with the outputs from the experts.
        #
        # 3. For each expert, process all tokens assigned to it (as determined by `flat_topk_idx == i`).
        #      Note: Each position in `flat_topk_idx` tells which expert that slot in `x` should go to. 
        #      We use boolean indexing: `x[flat_topk_idx == i]` gets inputs for expert i, and their result goes to the same locations in `y`.
        #      `.to(y.dtype)` makes sure the output dtype matches `y` (sometimes float16/float32).
        #
        # 4. After computing outputs for all experts and populating `y`, we now have (batch*seq*num_experts_per_tok, hidden)
        #      But we want to combine, for each token, its multiple expert outputs, weighted by the gate probabilities.
        #      - Reshape `y` to shape (batch*seq, num_experts_per_tok, hidden)
        #      - Multiply by topk_weight (shape batch*seq, num_experts_per_tok, 1) to scale outputs by gate's probability for each expert.
        #      - Sum over the experts (dim=1) to get the final combined output for each token.
        #
        # 5. Reshape back from flat (batch*seq, hidden) to original input shape (batch, seq, hidden).
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.mode_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        Efficient inference pass for Mixture-of-Experts (MoE) routing.

        Args:
            x (Tensor): Input tensor of shape (num_tokens, hidden_dim).
            flat_expert_indices (Tensor): 1D tensor, for each repeated token, gives its assigned expert index (length = num_tokens * num_experts_per_token).
            flat_expert_weights (Tensor): 1D tensor with the expert-assigned weights for each input (same length as flat_expert_indices).

        Returns:
            Tensor: Aggregated output tensor of shape (num_tokens, hidden_dim).
        """
        # Create an output tensor filled with zeros; this will accumulate the expert outputs for each token.
        expert_cache = torch.zeros_like(x)

        # Sort the expert assignments so that tokens for each expert are contiguous.
        idxs = flat_expert_indices.argsort()

        # For each expert, count how many tokens are assigned (including shared tokens for TopK > 1),
        # then compute the cumsum to get the end index of each expert's token region.
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # Since each original token is repeated num_experts_per_token times, recover
        # the original token indices. E.g., if token 5 is assigned to 2 experts, it appears twice,
        # and both repeated positions' integer division by num_experts_per_tok yields 5.
        token_idxs = idxs // self.config.num_experts_per_tok

        # ----
        # Example for intuition:
        # - Let tokens_per_expert = [6, 15, 20, 26], which means:
        #       * Expert 0 handles tokens idxs[:6]
        #       * Expert 1 handles tokens idxs[6:15]
        #       * etc.
        # - For each such window, token_idxs gives us which original tokens to process.
        # ----

        # Now, iterate over all experts and route their tokens.
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # No tokens assigned to this expert

            expert = self.experts[i]          # Get expert module
            exp_token_idx = token_idxs[start_idx:end_idx]  # Indices of tokens processed by this expert
            expert_tokens = x[exp_token_idx]  # Select tokens for this expert

            # Forward expert tokens through the expert network.
            expert_out = expert(expert_tokens).to(expert_cache.dtype)

            # Multiply each output by its assigned gating probability (weight)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # Add these outputs back to the cache at the right location.
            # Each token may have several expert outputs summed, so we accumulate.
            # The scatter_add_ below means: for each row in exp_token_idx, add the corresponding row
            # in expert_out to expert_cache at [exp_token_idx, :]
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        # The result is the weighted, summed expert outputs for each token.
        return expert_cache

"""
          Output
            │
       ┌────────┐
       │ Linear │
       └────────┘
            │
       ┌────────┐        ┌──────┐
       │ Linear │        │ SiLU |
       └────────┘        └──────┘
             │              │
             └─────x────────┘
                   │
                   |
             ┌──────────┐
             │  Linear  │
             └──────────┘
                   │
                 Input
"""
# Formula: output = dropout(down_proj(act_fn(gate_proj(x)) * up_proj(x)))

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x: [batch, seq, hidden_size]
        z1 = self.gate_proj(x)
        z2 = self.up_proj(x)
        a = self.act_fn(z1)
        h = a * z2
        y = self.down_proj(h)
        return self.dropout(y)

class MOEGate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(bsz * seq_len, h) # (-1, h)
        logits = F.linear(hidden_states, self.weight, None) # (-1, n_routed_experts)

        if self.scoring_func == "softmax":
            # scores is [num_tokens, n_routed_experts]: each row sums to 1 for a token over experts
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"Scoring function {self.scoring_func} not implemented")
        
        # Example:
        #   If scores for one token are [0.1, 0.7, 0.05, 0.15] and self.top_k=2,
        #   torch.topk(scores, 2) -> (tensor([0.7, 0.15]), tensor([1, 3]))
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)

        # If more than one expert per token and probability normalization is set,
        # re-normalize the top-k weights for each token so their sum is 1.
        # This can correct for imperfect softmax due to masking/etc.
        # For example, if topk_weight is tensor([0.7, 0.15]), after normalization it becomes tensor([0.8235, 0.1765]).
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # Epsilon prevents divide by zero
            topk_weight = topk_weight / denominator
        
        if self.training and self.norm_topk_prob:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # The aux_loss is computed as:
                #   aux_loss = α * mean_b[ sum_i( E_i * P̂_i ) ]
                # where:
                #   α: self.alpha (auxiliary loss scaling factor)
                #   E_i: fraction of times expert i was assigned in the sequence (for each batch)
                #   P̂_i: mean probability assigned to expert i over the sequence (for each batch)
                #   mean_b: average over batches
                # Concretely:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # shape: (bsz, seq_len, n_routed_experts)
                # ce[b, i]: fraction of times expert i was assigned in batch b
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Explanation:
                # This block computes an auxiliary load balancing loss (Shazeer et al. 2017 style)
                # to encourage tokens to be distributed more evenly among experts.
                #
                # 1. mask_ce: Make a (num_tokens * top_k, n_routed_experts) one-hot tensor where each row
                #    indicates which expert was selected for each position.
                # 2. ce: Fractional assignment to each expert across all tokens (probability a token chooses each expert).
                #    This is the average number of times each expert was chosen.
                # 3. Pi: Average (softmax) routing probability given to each expert by the gating network.
                # 4. fi: Normalize assignment fraction by multiplying by n_routed_experts, gives "usage".
                # 5. aux_loss: Dot product of Pi and fi, summed and scaled by alpha. 
                #    This penalizes load imbalance by rewarding even assignment and even probabilities.
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)  # [num_tokens * top_k, n_experts]
                ce = mask_ce.float().mean(0)  # Fraction of tokens assigned to each expert: [n_experts]
                Pi = scores_for_aux.mean(0)   # Average probability for each expert over all tokens: [n_experts]
                fi = ce * self.n_routed_experts    # n_experts * assignment fraction ("usage" per expert): [n_experts]
                aux_loss = (Pi * fi).sum() * self.alpha  # Scalar: scaling coefficient times the load balancing loss
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss
