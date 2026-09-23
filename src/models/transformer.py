import torch
import torch.nn as nn
import torch.nn.functional as F

current_device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Causal self-attention head
# =========================

class Head(nn.Module):
    def __init__(self, emb_dim, head_size):
        super().__init__()

        self.key = nn.Linear(
            emb_dim, head_size, bias=False, device=current_device
        )
        self.querry = nn.Linear(
            emb_dim, head_size, bias=False, device=current_device
        )
        self.value = nn.Linear(
            emb_dim, head_size, bias=False, device=current_device
        )

    def forward(self, x):

        k = self.key(x)
        q = self.querry(x)
        v = self.value(x)

        out =  F.scaled_dot_product_attention(q,
                                              k,
                                              v,
                                              is_causal=True)

        return out

    
#Tu skorzystać z metody torcha

# =========================
# Multi-head self-attention
# =========================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(emb_dim, head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            emb_dim, emb_dim, device=current_device
        )

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.projection(out)


# =========================
# Feed-forward network
# =========================

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                emb_dim,
                4 * emb_dim,
                device=current_device
            ),
            nn.ReLU(),
            nn.Linear(
                4 * emb_dim,
                emb_dim,
                device=current_device
            ),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.network(x)


# =========================
# Transformer decoder block
# =========================

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, emb_dim):
        super().__init__()

        head_size = emb_dim // num_heads

        self.sa = MultiHeadSelfAttention(
            num_heads,
            emb_dim,
            head_size
        )

        self.ffwd = FeedForward(emb_dim)

        self.ln1 = nn.LayerNorm(
            emb_dim,
            device=current_device
        )

        self.ln2 = nn.LayerNorm(
            emb_dim,
            device=current_device
        )

    def forward(self, x):

        # Pre-norm residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# =========================
# Autoregressive Transformer
# =========================

class Transformer_MODEL(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        num_of_spins,
        n_layers
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_spins = num_of_spins

        # Tokens: -1 -> 0, +1 -> 1, START -> 2
        self.token_embedding = nn.Embedding(
            3,
            emb_dim,
            device=current_device
        )

        self.position_embedding = nn.Embedding(
            num_of_spins + 1,
            emb_dim,
            device=current_device
        )

        layers = []

        for _ in range(n_layers):
            layers.append(
                DecoderBlock(num_heads, emb_dim)
            )

        self.network = nn.Sequential(*layers)

        self.ln_f = nn.LayerNorm(
            emb_dim,
            device=current_device
        )

        # Predicts the probability of the next spin
        self.output = nn.Linear(
            emb_dim,
            1,
            device=current_device
        )

    def sample(self, batch_size):

        generated_spin_values = torch.zeros(
            batch_size,
            self.num_spins,
            dtype=torch.long,
            device=current_device
        )

        log_prob_system = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=current_device
        )

        start_token_id = 2

        # Generate spins autoregressively
        for i in range(self.num_spins):

            if i == 0:
                input_tokens = torch.full(
                    (batch_size, 1),
                    start_token_id,
                    device=current_device
                )

            else:
                # Convert spins {-1, +1} to token IDs {0, 1}
                seq_of_tokens = (
                    (generated_spin_values[:, :i] + 1) // 2
                ).long()

                starting_token = torch.full(
                    (batch_size, 1),
                    start_token_id,
                    device=current_device
                )

                input_tokens = torch.cat(
                    (starting_token, seq_of_tokens),
                    dim=1
                )

            # Add token and positional embeddings
            current_positions = torch.arange(
                input_tokens.shape[1],
                device=current_device
            )

            input_emb = (
                self.token_embedding(input_tokens)
                + self.position_embedding(current_positions)
            )

            # Transformer decoder forward pass
            transf_output = self.network(input_emb)
            norm_output = self.ln_f(transf_output)

            # Use the last token representation to predict the next spin
            last_arg = norm_output[:, -1, :]

            logit = self.output(last_arg).squeeze(-1)
            spin_prob = torch.sigmoid(logit)

            # Bernoulli sample: 0 -> -1, 1 -> +1
            spin = (
                torch.bernoulli(spin_prob).detach() * 2 - 1
            )

            generated_spin_values[:, i] = spin

            # Accumulate log-probability of the generated configuration
            log_prob_system += torch.where(
                spin == 1,
                torch.log(spin_prob + 1e-8),
                torch.log(1 - spin_prob + 1e-8)
            )

        return generated_spin_values, log_prob_system