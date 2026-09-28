from src.metrics.metrics import load_config
from src.trainer.trainer import Trainer
from src.metrics.metrics import evaluate_model
from src.metrics.metrics import save_result
from src.metrics.metrics import Model_Storage
from src.visualization.plots import save_plots_same_model
import torch
import gc
from pathlib import Path
import random
import numpy as np


def set_seed(seed=235):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_lstm():

    set_seed(7)

    #Setting paths
    batch_size128 = Path("experiments/configs/Lstms/batch_size128.yaml")
    batch_size512 = Path("experiments/configs/Lstms/batch_size512.yaml")
    batch_size1024 = Path("experiments/configs/Lstms/batch_size1024.yaml")

    learning_rate01 = Path("experiments/configs/Lstms/learning_rate01.yaml")
    learning_rate001 = Path("experiments/configs/Lstms/learning_rate001.yaml")
    learning_rate0001 = Path("experiments/configs/Lstms/learning_rate0001.yaml")

    spin_info_false = Path("experiments/configs/Lstms/spin_info_false.yaml")
    spin_info_true = Path("experiments/configs/Lstms/spin_info_true.yaml")

    path_table_learning_rate = [learning_rate01, learning_rate001, learning_rate0001]
    path_table_batch_size = [batch_size128, batch_size512, batch_size1024]
    path_table_spin_info = [spin_info_false, spin_info_true]

    hiperparameters_table = [path_table_batch_size, path_table_spin_info, path_table_learning_rate]
    hiperparameters_names = ["Batch size (lstm)", "Spin Information (lstm)", "Learning rate (lstm)"]
    
    model_names =[['batch_size128', 'batch_size512', 'batch_size1024'], 
                  ['spin_info_false', 'spin_info_true'], 
                  ['learning_rate01', 'learning_rate001', 'learning_rate0001']]

    """===LSTM==="""
    i=0
    for path_table in hiperparameters_table:
        data_table=[]
        k=0
        for path in path_table:
            print("---Initializing LSTM model ---")

            config_lstm = load_config(path)
            model = Trainer(model=config_lstm['model'],
                            n=config_lstm['lattice_size'],
                            beta =config_lstm['beta'],
                            epochs=config_lstm['training']['epochs'],
                            lr=config_lstm['training']['learning_rate'],
                            batch_size=config_lstm['training']['batch_size'],
                            spin_placement_info=config_lstm['model_params']['spin_placement_info'],
                            n_layers=1,
                            hiperparam_experiment = True)
            
            print("--- Training start ---")
            model.train_model()
            print("--- Training complete ---")
            
            print("--- Saving results ---")
            results = evaluate_model(model, hiperparam_tested=model_names[i][k])
            save_result(results, "hiperparam_test", "LSTM")

            data = Model_Storage(model)
            data_table.append(data)
            del results
            del model
            del config_lstm
            del data
            gc.collect()
            torch.cuda.empty_cache()
            k+=1
        
        save_plots_same_model(data_table, folder = "hiperparam_test", hiperparam_name=hiperparameters_names[i], model_names=model_names[i])
        i+=1

        del data_table

if __name__ == "__main__":
    run_lstm()