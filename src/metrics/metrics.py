from src.physics.energy import calculate_variational_free_energy
import numpy as np
import torch
import time
import csv
import yaml

"""Jakość przybliżenia rozkładu"""
def sample_vfe(model):
    """Return the mean VFE per spin over the newly sampled batch"""
    sampled_spins, log_prob = model.model.sample(model.batch_size)
    return calculate_variational_free_energy(model.beta, log_prob, sampled_spins, model.batch_size, model.n).mean() / (model.n**2)

def compare_analitycal(model, nb_batches):
    """Compares mean VFE over nb_batches with analitical value"""
    analitycal_value = -2.6359026301137902

    estimates=[]
    for _ in range(nb_batches):
        estimates.append(sample_vfe(model).item())

    estimates = np.array(estimates)

    mean_vfe = estimates.mean()
    std_vfe = estimates.std()

    abs_error = abs(mean_vfe - analitycal_value)

    return {
        "mean_vfe": mean_vfe,
        "std_vfe": std_vfe,
        "absolute_error": abs_error
    }

def sample_ESS(model):
    """Calculates ESS of sample"""
    sampled_spins, log_prob = model.model.sample(model.batch_size)

    vfe = calculate_variational_free_energy(
            model.beta,
            log_prob,
            sampled_spins,
            model.batch_size,
            model.n
            )
    loss_per_sample = model.beta * vfe

    ESS_energy = torch.exp(
        2 * torch.logsumexp(
                            -loss_per_sample,
                            dim=0
                            )
        - torch.logsumexp(
                            -2 * loss_per_sample,
                            dim=0
                            )
                        ) / model.batch_size

    return ESS_energy

def ESS_info(model, nb_batches):
    """Return dict mean ESS and its std over few(nb_batches) samples"""
    estimates = []
    for _ in range(nb_batches):
        estimates.append(sample_ESS(model))

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

def generating_samples_time(model, nb_samples):
    """Returns time to generate given number of samples"""
    start_time = time.time()
    for _ in range(nb_samples):
        sampled_spins, log_prob = model.model.sample(model.batch_size)

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

"""Evaluation"""
def evaluate_model(model):
    vfe_metrics = compare_analitycal(model, 100)

    evaluation = {
        "Mean_vfe": vfe_metrics['mean_vfe'],
        "Std_vfe": vfe_metrics['std_vfe'],
        "VFE_abs_err": vfe_metrics['absolute_error'],
        "ESS": ESS_info(model, 100),
        "Nb_of_params": number_of_params(model),
        "Time_to_generate_samples": generating_samples_time(model, 10000),
        "Training epochs": convergence(model)
    }
    return evaluation

def save_result(evaluation: dict, model_name: str):
    
    with open(f"results/tables/{model_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=evaluation.keys())
        w.writeheader()
        w.writerow(evaluation)


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config