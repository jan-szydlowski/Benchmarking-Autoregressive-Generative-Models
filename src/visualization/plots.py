import matplotlib.pyplot as plt
from src.metrics.metrics import get_vfe_mean_values

def plot_mean_vfe_training(*Models, model_name: list = None):
    if Models[0][0].hiperparam_experiment:
        for i in range(len(Models[0])):
            plt.plot(get_vfe_mean_values(Models[0][i]), label=f"{model_name[i]}")
    else:
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


def plot_ESS_training(*Models, model_name: list = None):
    if Models[0][0].hiperparam_experiment:
        for i in range(len(Models[0])):
            ess_table = [t.cpu() for t in Models[0][i].ess_arr]
            plt.plot(ess_table, label=f"{model_name[i]}")
    else:
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



def save_plots_same_model(Models: list, folder: str, hiperparam_name: str, model_names: str):
    plot_mean_vfe_training(Models, model_name=model_names)
    plt.savefig(f'results/figures/{folder}/{hiperparam_name}_vfe.png')
    plt.close()
    plot_ESS_training(Models, model_name=model_names)
    plt.savefig(f'results/figures/{folder}/{hiperparam_name}_ESS.png')
    plt.close()

def save_plots_diff_model(*Models, folder: str):
    plot_mean_vfe_training(Models)
    plt.savefig(f'results/figures/{folder}/models_VFE.png')
    plt.close()
    plot_ESS_training(Models)
    plt.savefig(f'results/figures/{folder}/models_ESS.png')
    plt.close()
    plot_energy(Models)
    plt.savefig(f'results/figures/{folder}/models_Energy.png')
    plt.close()
    plot_magnetization(Models)
    plt.savefig(f'results/figures/{folder}/models_Magnetization.png')
    plt.close()