import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Custom LSTM implementation
# =========================

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # Input gate
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Candidate memory
        self.W_g = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_g = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

        # Output gate
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def lstm_unit(self, input_value, long_memory, short_memory):

        # Forget gate
        percent_to_remember = torch.sigmoid(
            input_value @ self.W_f.T
            + short_memory @ self.U_f.T
            + self.b_f
        )

        # Candidate memory and input gate
        long_term_change = torch.tanh(
            input_value @ self.W_i.T
            + short_memory @ self.U_i.T
            + self.b_i
        )

        how_much_long_term_change = torch.sigmoid(
            input_value @ self.W_g.T
            + short_memory @ self.U_g.T
            + self.b_g
        )

        # Update cell state
        long_memory_updated = (
            long_memory * percent_to_remember
            + long_term_change * how_much_long_term_change
        )

        # Output gate
        how_much_short_term_update = torch.sigmoid(
            input_value @ self.W_o.T
            + short_memory @ self.U_o.T
            + self.b_o
        )

        # Update hidden state
        short_memory_updated = (
            torch.tanh(long_memory_updated)
            * how_much_short_term_update
        )

        return long_memory_updated, short_memory_updated

    def forward(self, input):

        batch_size = input.shape[0]
        seq_len = input.shape[1]

        # Initial hidden and cell states
        long_memory = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        short_memory = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device
        )

        outputs = []

        for i in range(seq_len):
            input_i = input[:, i, :]

            long_memory, short_memory = self.lstm_unit(
                input_i,
                long_memory,
                short_memory
            )

            outputs.append(short_memory)

        return torch.stack(outputs, dim=1)


# =========================
# LSTM autoregressive model
# =========================

class ExtractLSTMOutput(nn.Module):
    def forward(self, input):
        # nn.LSTM returns: output, (h_n, c_n)
        return input[0]


class LSTM_MODEL(nn.Module):
    def __init__(self, n, spin_placement_info):
        super().__init__()

        self.n_lattice = n
        self.num_spins = n * n
        self.hidden_size = 45
        self.spin_placement_info = spin_placement_info

        input_size = 3 if spin_placement_info else 1

        self.network = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                batch_first=True
            ),
            ExtractLSTMOutput()
        )

        # Predicts the probability of the next spin
        self.output_layer = nn.Linear(self.hidden_size, 1)

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

            if i == 0:
                # Start token
                lstm_input = torch.zeros(
                    batch_size,
                    1,
                    1,
                    device=device
                )
            else:
                # Previously generated spins
                lstm_input = generated_spins[:, :i].unsqueeze(-1)

            if self.spin_placement_info:

                # Add normalized lattice coordinates
                row = i // self.n_lattice
                col = i % self.n_lattice

                row_norm = row / (self.n_lattice - 1)
                col_norm = col / (self.n_lattice - 1)

                seq_len_lstm = lstm_input.shape[1]

                row_feature = torch.full(
                    (batch_size, seq_len_lstm, 1),
                    row_norm,
                    device=device
                )

                col_feature = torch.full(
                    (batch_size, seq_len_lstm, 1),
                    col_norm,
                    device=device
                )

                current_input = torch.cat(
                    [lstm_input, row_feature, col_feature],
                    dim=2,
                )

            else:
                current_input = lstm_input

            lstm_output = self.network(current_input)

            # Hidden state corresponding to the last sequence element
            last_hidden = lstm_output[:, -1, :]

            current_logits = self.output_layer(
                last_hidden
            ).squeeze(-1)

            prob_current_spin = torch.sigmoid(current_logits)

            # Bernoulli sample: 0 -> -1, 1 -> +1
            sampled_bit = torch.bernoulli(
                prob_current_spin
            ).detach()

            spin = sampled_bit * 2 - 1

            # Store newly generated spin
            next_generated_spins = generated_spins.clone()
            next_generated_spins[:, i] = spin
            generated_spins = next_generated_spins

            # Accumulate log-probability of the generated configuration
            log_prob_system += torch.where(
                sampled_bit == 1,
                torch.log(prob_current_spin + 1e-8),
                torch.log(1 - prob_current_spin + 1e-8)
            )

        return generated_spins, log_prob_system