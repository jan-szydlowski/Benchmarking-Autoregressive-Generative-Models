import torch
from torch.optim import Adam
from src.physics.energy import calculate_variational_free_energy, calculate_Ising_energy
import time
import numpy as np
from src.models.transformer import Transformer_MODEL
from src.models.van import VAN_MODEL
from src.models.pixelcnn import PixelCNN_MODEL
from src.models.lstm import LSTM_MODEL



class Trainer():
    def __init__(
        self,
        n,
        beta,
        epochs,
        lr,
        batch_size,
        model,
        n_layers,
        **Model_params
    ):
        """
        Trainer for the autoregressive Transformer model.
        """

        self.n = n
        self.num_spins = n*n
        self.n_layers = n_layers
        self.epochs = epochs
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.model_name = model

        if model == "transformer":
            self.model = Transformer_MODEL(
                emb_dim=Model_params['embedding_dim'],
                num_heads=Model_params['num_heads'],
                num_of_spins=n*n,
                n_layers=n_layers
            )

        if model == "van":
            self.model = VAN_MODEL(n=n, 
                               nb_layers=self.n_layers
                               )
            
        if model == "pixelcnn":
            self.model = PixelCNN_MODEL(n=n, 
                               kernel_size=Model_params['kernel_size'], 
                               num_of_layers=self.n_layers, 
                               channels=Model_params['channels']
                               )
            
        if model == "lstm":
            self.model = LSTM_MODEL(n=n, 
                               spin_placement_info=Model_params['spin_placement_info']
                               )


    def train_model(self):
        start_time = time.time()
        self.magnetization = []
        self.vfe_mean = []
        self.ess_arr = []
        self.energy_per_spin_arr = []

        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr
        )

        for epoch in range(self.epochs):

            optimizer.zero_grad()

            # Generate samples and their log probabilities
            sampled_spins, log_prob = self.model.sample(
                self.batch_size
            )

            # VFE acts only as the REINFORCE signal
            with torch.no_grad():

                vfe = calculate_variational_free_energy(
                    self.beta,
                    log_prob,
                    sampled_spins,
                    self.batch_size,
                    self.n
                )

                energy = calculate_Ising_energy(
                    self.n,
                    self.batch_size,
                    sampled_spins
                )

                energy_per_spin = energy / (self.n * self.n)

            self.energy_per_spin_arr.append(
                energy_per_spin.mean()
            )

            self.magnetization.append(
                np.array(sampled_spins.cpu()).sum()
            )

            self.vfe_mean.append(
                vfe.mean()
            )

            loss_per_sample = self.beta * vfe

            # Mean subtraction is used as a baseline
            # to reduce REINFORCE gradient variance
            loss_reinforce = (
                (loss_per_sample - loss_per_sample.mean())
                * log_prob
            ).mean()

            loss_reinforce.backward()
            optimizer.step()

            with torch.no_grad():

                # Normalized Effective Sample Size
                # calculated in log-space for numerical stability
                ESS_energy = torch.exp(
                    2 * torch.logsumexp(
                        -loss_per_sample,
                        dim=0
                    )
                    - torch.logsumexp(
                        -2 * loss_per_sample,
                        dim=0
                    )
                ) / self.batch_size

            self.ess_arr.append(
                ESS_energy
            )

            if (epoch + 1) % 1 == 0:
                print(
                    f"Epoch {epoch+1}, "
                    f"Loss: {loss_reinforce.item():.4f}, "
                    f"ESS_energy: {ESS_energy.item():.4f}, "
                    f"VFE: {vfe.mean().item():.4f}"
                )
            if abs(vfe.mean().item() - 2.6359026301137902) < 0.01:
                break
        self.last_epoch = epoch    
        self.end_time = (time.time() - start_time)