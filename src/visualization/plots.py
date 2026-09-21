import matplotlib.pyplot as plt
from src.metrics.metricsv2 import get_vfe_mean_values

def plot_mean_vfe_training(*Models):
    for i in range(len(Models)):
        plt.plot(get_vfe_mean_values(Models[i]), label=f"Model {i+1}")
    plt.axhline(
        y=-2.6359026301137902,
        color="black",
        linestyle="--",
        label="Analytical value"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean VFE per spin")
    plt.legend()
    plt.show()

def plot_ESS_training(*Models):
    for i in range(len(Models)):
        plt.plot(Models[i].ess_arr, label=f"Model {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("ESS value")
    plt.legend()
    plt.show()

#Plot energy i magneyzacja