import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RewardModel(nn.Module):
    """Siamese reward model: one shared-weight encoder scores (prompt, response)
    pairs. A and B sides of a batch are interleaved as [a0, b0, a1, b1, ...]
    (see dataset.RewardPairCollator) and run through the encoder in a single
    forward pass, then split back apart here.

    Bradley-Terry, Variant A (learnable threshold delta):
        d = r_a - r_b
        P(a wins) = sigmoid(d - delta)
        P(b wins) = sigmoid(-d - delta)
        P(tie)    = 1 - P(a wins) - P(b wins)
    These three probabilities are turned into logits (via log) and trained
    with standard 3-class cross-entropy, not a margin-ranking loss.
    """

    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-base",
        pooling: str = "cls",
        dropout: float = 0.1,
        delta_init: float = 0.0,
    ):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.pooling = pooling

        config = AutoConfig.from_pretrained(backbone)
        # Some checkpoints (e.g. microsoft/deberta-v3-base) set torch_dtype
        # in config.json to float16; from_pretrained would otherwise load
        # half-precision weights by default and clash with the fp32
        # reward_head below. fp16 training is handled by autocast in
        # train.py, so weights are always loaded in float32 here.
        self.encoder = AutoModel.from_pretrained(backbone, config=config, torch_dtype=torch.float32)

        self.dropout = nn.Dropout(dropout)
        self.reward_head = nn.Linear(config.hidden_size, 1)

        # Bradley-Terry Variant A: learnable tie threshold.
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))

        # Small eps to keep log() numerically safe when a probability
        # rounds to exactly 0 under fp16 autocast.
        self._eps = 1e-6

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]

        # mean pooling over real (non-padding) tokens
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counted

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns a scalar reward per input row: shape (N,)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.reward_head(pooled).squeeze(-1)  # (N,)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """input_ids / attention_mask: (batch * 2, seq_len), interleaved [a0, b0, a1, b1, ...].

        Returns a dict with r_a, r_b (batch,), d = r_a - r_b, delta, and
        class_logits (batch, 3) in [a, b, tie] order, ready for
        nn.functional.cross_entropy against labels encoded the same way
        (see dataset.WINNER_TO_LABEL).
        """
        rewards = self.encode(input_ids, attention_mask)  # (batch * 2,)
        r_a = rewards[0::2]
        r_b = rewards[1::2]

        d = r_a - r_b

        p_a = torch.sigmoid(d - self.delta)
        p_b = torch.sigmoid(-d - self.delta)
        p_tie = (1.0 - p_a - p_b).clamp(min=self._eps)

        # cross_entropy expects logits, not probabilities: log(p) works here
        # because cross_entropy(logits) = -log(softmax(logits)) and we want
        # -log(p) directly, i.e. log(p) as an unnormalized logit is fine
        # since softmax(log(p)) == p when p already sums to 1.
        class_logits = torch.stack(
            [torch.log(p_a.clamp(min=self._eps)),
             torch.log(p_b.clamp(min=self._eps)),
             torch.log(p_tie)],
            dim=1,
        )  # (batch, 3)

        return {
            "r_a": r_a,
            "r_b": r_b,
            "d": d,
            "delta": self.delta,
            "p_a": p_a,
            "p_b": p_b,
            "p_tie": p_tie,
            "class_logits": class_logits,
        }