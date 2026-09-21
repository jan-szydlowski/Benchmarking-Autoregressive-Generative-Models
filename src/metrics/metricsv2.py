from src.physics.energy import calculate_variational_free_energy, calculate_Ising_energy
import numpy as np
import torch
import time

"""Jakość przybliżenia rozkładu"""
def sample_vfe(model, batch_size, n):
    """Return the mean VFE per spin over the newly sampled batch"""
    sampled_spins, log_prob = model.sample(batch_size)
    return calculate_variational_free_energy(log_prob, sampled_spins).mean() / (n**2)

def compare_analitycal(model, batch_size, n, nb_batches):
    """Compares mean VFE over nb_batches with analitical value"""
    analitycal_value = -2.6359026301137902

    estimates=[]
    for _ in range(nb_batches):
        estimates.append(sample_vfe(model, batch_size, n).item())

    estimates = np.array(estimates)

    mean_vfe = estimates.mean()
    std_vfe = estimates.std()

    abs_error = abs(mean_vfe - analitycal_value)

    return {
        "mean_vfe": mean_vfe,
        "std_vfe": std_vfe,
        "absolute_error": abs_error
    }

def sample_ESS(model, batch_size, beta):
    """Calculates ESS of sample"""
    sampled_spins, log_prob = model.sample(batch_size)

    vfe = calculate_variational_free_energy(
            beta,
            log_prob,
            sampled_spins
            )
    loss_per_sample = beta * vfe

    ESS_energy = torch.exp(
        2 * torch.logsumexp(
                            -loss_per_sample,
                            dim=0
                            )
        - torch.logsumexp(
                            -2 * loss_per_sample,
                            dim=0
                            )
                        ) / batch_size

    return ESS_energy

def ESS_info(model, batch_size, beta, nb_batches):
    """Return dict mean ESS and its std over few(nb_batches) samples"""
    estimates = []
    for _ in range(nb_batches):
        estimates.append(sample_ESS(model, batch_size, beta))

    estimates = np.array(estimates)

    mean_ESS = estimates.mean()
    std_ESS = estimates.std()

    return {
        "mean ess": mean_ESS,
        "std ess": std_ESS
    }

def convergence(model):
    """Returns models epoch after succesfuly learning distribution"""
    return model.last_epoch()

def training_time(model):
    """Return training time"""
    return model.learning_time()

def generating_samples_time(model, batch_size, nb_samples):
    """Returns time to generate given number of samples"""
    start_time = time.time()
    for _ in range(nb_samples):
        sampled_spins, log_prob = model.sample(batch_size)

    end_time = (time.time() - start_time)
    print(f"---Generating {nb_samples} samples took {end_time} seconds ---")

    return None

def number_of_params(model):
    """Return the number of all parameters"""
    return sum(p.numel() for p in model.parameters())


"""#Metrics For Plots"""
def get_vfe_mean_values(model):
    """Returns mean VFE per spin values per epoch"""
    return torch.tensor(model.vfe_mean) / (model.n * model.n)

