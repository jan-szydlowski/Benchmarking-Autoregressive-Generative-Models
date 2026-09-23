import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Masked linear layer
# =========================

class MaskedLinear(nn.Module):
    def __init__(self, n_flat, first_layer=False):
        super().__init__()

        self.linear = nn.Linear(n_flat, n_flat)

        # First layer cannot access the current or future spins
        if first_layer:
            mask = torch.tril(
                torch.ones(n_flat, n_flat),
                diagonal=-1
            )

        # Hidden layers can access the current representation
        else:
            mask = torch.tril(
                torch.ones(n_flat, n_flat),
                diagonal=0
            )

        self.register_buffer("mask", mask)

    def forward(self, x):

        # Apply autoregressive mask to the weights
        weights = self.linear.weight * self.mask

        return F.linear(
            x,
            weights,
            self.linear.bias
        )


# =========================
# Variational Autoregressive Network
# =========================

class VAN_MODEL(nn.Module):
    def __init__(self, n, nb_layers):
        super().__init__()

        self.n_lattice = n
        self.num_spins = n * n

        layers = [
            MaskedLinear(
                n_flat=self.num_spins,
                first_layer=True
            )
        ]

        for _ in range(nb_layers - 1):
            layers.append(
                nn.LeakyReLU(0.05)
            )

            layers.append(
                MaskedLinear(
                    n_flat=self.num_spins,
                    first_layer=False
                )
            )

        self.network = nn.Sequential(*layers)

    def sample(self, batch_size):
        device = next(self.parameters()).device
        
        generated_spins = torch.zeros(
            batch_size,
            self.num_spins,
            device=device
        )

        log_prob_system = torch.zeros(
            batch_size,
            device=device
        )

        # Generate spins autoregressively
        for i in range(self.num_spins):

            logits_all = self.network(
                generated_spins
            )

            current_logits = logits_all[:, i]

            prob_current_spin = torch.sigmoid(
                current_logits
            )

            # Bernoulli sample: 0 -> -1, 1 -> +1
            spin = (
                torch.bernoulli(prob_current_spin) * 2 - 1
            ).detach()

            # Accumulate log-probability of the generated configuration
            log_prob_system += torch.where(
                spin == 1,
                torch.log(prob_current_spin + 1e-8),
                torch.log(1 - prob_current_spin + 1e-8)
            )

            # Store newly generated spin
            next_generated_spins = generated_spins.clone()
            next_generated_spins[:, i] = spin
            generated_spins = next_generated_spins

        return generated_spins, log_prob_system