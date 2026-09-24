import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Masked convolution layer
# =========================

class MaskedCNN(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _, _, height, width = self.weight.shape

        self.register_buffer(
            "mask",
            torch.ones_like(self.weight)
        )

        # Type A mask: excludes the current pixel
        if mask_type == "A":
            self.mask[:, :, height // 2:, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

        # Type B mask: includes the current pixel
        if mask_type == "B":
            self.mask[:, :, height // 2:, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

        # Apply the mask to initialized weights
        self.weight.data *= self.mask

        # Compensate for reduced number of active connections
        self.weight.data *= torch.sqrt(
            self.mask.numel() / self.mask.sum()
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


# =========================
# PixelCNN autoregressive model
# =========================

class PixelCNN_MODEL(nn.Module):
    def __init__(self, n, kernel_size, num_of_layers, channels):
        super().__init__()

        self.num_spins = n * n
        self.n = n

        layers = []

        for i in range(num_of_layers):

            if i == 0:
                # First layer must not access the current spin
                layers.append(
                    MaskedCNN(
                        "A",
                        in_channels=1,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )

                layers.append(
                    nn.PReLU(channels, init=0.5)
                )

            else:
                # Hidden layers can access the current intermediate representation
                layers.append(
                    MaskedCNN(
                        "B",
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )

                layers.append(
                    nn.PReLU(channels, init=0.5)
                )

        self.network = nn.Sequential(*layers)

        # Maps hidden representation to one output logit per spin
        self.output = MaskedCNN(
            "B",
            in_channels=channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

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

        # Generate spins sequentially in raster-scan order
        for i in range(self.num_spins):

            conv2d_input = generated_spins.clone().view(
                batch_size,
                1,
                self.n,
                self.n
            )

            row = i // self.n
            col = i % self.n

            logits_all = self.output(
                self.network(conv2d_input)
            )

            current_logits = logits_all[:, 0, row, col]

            prob_current_spin = torch.sigmoid(
                current_logits
            )

            # Bernoulli sample: 0 -> -1, 1 -> +1
            spin = torch.bernoulli(
                prob_current_spin
            ) * 2 - 1

            spin = spin.detach()

            # Accumulate log-probability of the generated configuration
            log_prob_system += torch.where(
                spin == 1,
                torch.log(prob_current_spin + 1e-8),
                torch.log(1 - prob_current_spin + 1e-8)
            )

            generated_spins[:, i] = spin

        return generated_spins, log_prob_system