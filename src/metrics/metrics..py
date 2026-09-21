import torch
import matplotlib.pyplot as plt
from src.physics.energy import calculate_variational_free_energy, calculate_Ising_energy

"""Sample from trained model"""
def sample_vfe(model, batch_size, n):
    sampled_spins, log_prob = model.sample(batch_size)
    # Return the mean VFE per spin over the newly sampled batch
    return calculate_variational_free_energy(log_prob, sampled_spins).mean() / (n**2)


def sample_energy(model, batch_size, n):
    sampled_spins, _ = model.sample(batch_size)
    # Return the mean energy per spin over the newly sampled batch
    return calculate_Ising_energy(sampled_spins).mean() / (n**2)


def sample_magnetization(model):
    sampled_spins, _ = model.sample(model.batch_size)
    # Return the mean magnetization over the newly sampled batch
    return torch.mean(sampled_spins).detach()

"""Get values from training"""
def get_mean_vfe_per_spin(vfe_mean, n):
    return torch.tensor(vfe_mean) / (n * n)

def get_ESS_values(ess_arr):
    return ess_arr

def get_mean_energy_per_spin(energy_per_spin_arr, n):
    return torch.tensor(energy_per_spin_arr) / (n * n)

def get_model_params(n, beta, epochs, lr, batch_size, nb_layers):
    print(f"N: {n}, ",
              f"Beta: {beta}, ",
              f"Epochs: {epochs}, ",
              f"Lr: {lr}, "
              f"Batch size: {batch_size}, ",
              f"Number of hidden layers: {nb_layers}")


"""PLOTS"""
def get_vfe_mean(vfe_mean, n):
    """Generates plot of VFE per iteration with respect to analitical value"""
    plt.plot(torch.tensor(vfe_mean) / (n*n))
    plt.axhline(y = -2.6359026301137902, color='black')
    plt.xlabel("Iteration")
    plt.ylabel("Mean VFE for spin")
    plt.legend(["VFE" ,"Analitical Value"])

def ESS_plot(ess_arr):
    """Generates ESS plot"""
    plt.plot(ess_arr)
    plt.xlabel("Iteration")
    plt.ylabel("Efective Sample Size")
