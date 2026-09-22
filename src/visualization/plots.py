import matplotlib.pyplot as plt
from metrics.metrics import get_vfe_mean_values

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
    plt.ylabel("ESS")
    plt.legend()
    plt.show()


def plot_energy(*Models):
    for i in range(len(Models)):
        plt.plot(Models[i].energy_per_spin_arr, label=f"Model {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Energy per spin")
    plt.legend()
    plt.show()


def plot_magnetization(*Models):
    for i in range(len(Models)):
        plt.plot(Models[i].magnetization, label=f"Model {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()



def save_plots_same_model(*Models, model_name):
    plot_mean_vfe_training(Models)
    plt.savefig(f'results/figures/{model_name}_vfe.png')
    plot_ESS_training(Models)
    plt.savefig(f'results/figures/{model_name}_ESS.png')

def save_plots_diff_model(*Models):
    plot_mean_vfe_training(Models)
    plt.savefig(f'results/figures/arch_VFE.png')
    plot_ESS_training(Models)
    plt.savefig(f'results/figures/arch_ESS.png')
    plot_energy(Models)
    plt.savefig(f'results/figures/arch_Energy.png')
    plot_magnetization(Models)
    plt.savefig(f'results/figures/arch_Magnetization.png')