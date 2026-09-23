import matplotlib.pyplot as plt
from src.metrics.metrics import get_vfe_mean_values

def plot_mean_vfe_training(*Models):
    for i in range(len(Models[0])):
        plt.plot(get_vfe_mean_values(Models[0][i]), label=f"{Models[0][i].model_name}")
    plt.axhline(
        y=-2.6359026301137902,
        color="black",
        linestyle="--",
        label="Analytical value"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean VFE per spin")
    plt.title("History of mean VFE")
    plt.grid(visible=True)
    plt.legend()
    plt.show()


def plot_ESS_training(*Models):
    for i in range(len(Models[0])):
        ess_table = [t.cpu() for t in Models[0][i].ess_arr]
        plt.plot(ess_table, label=f"{Models[0][i].model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("ESS")
    plt.title("History of ESS")
    plt.grid(visible=True)
    plt.legend()
    plt.show()


def plot_energy(*Models):
    for i in range(len(Models[0])):
        energy_table = [t.cpu() for t in Models[0][i].energy_per_spin_arr]
        plt.plot(energy_table, label=f"{Models[0][i].model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Energy per spin")
    plt.title("History of energy")
    plt.grid(visible=True)
    plt.legend()
    plt.show()


def plot_magnetization(*Models):
    print(Models[0])
    print(Models)
    print(Models[0][0])
    for i in range(len(Models[0])):
        plt.plot(Models[0][i].magnetization, label=f"{Models[0][i].model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Magnetization")
    plt.title("History of magnetization")
    plt.grid(visible=True)
    plt.legend()
    plt.show()



def save_plots_same_model(*Models, model_name):
    plot_mean_vfe_training(Models)
    plt.savefig(f'results/figures/{model_name}_vfe.png')
    plt.close()
    plot_ESS_training(Models)
    plt.savefig(f'results/figures/{model_name}_ESS.png')
    plt.close()

def save_plots_diff_model(*Models):
    plot_mean_vfe_training(Models)
    plt.savefig(f'results/figures/models_VFE.png')
    plt.close()
    plot_ESS_training(Models)
    plt.savefig(f'results/figures/models_ESS.png')
    plt.close()
    plot_energy(Models)
    plt.savefig(f'results/figures/models_Energy.png')
    plt.close()
    plot_magnetization(Models)
    plt.savefig(f'results/figures/models_Magnetization.png')
    plt.close()