from src.metrics.metrics import load_config
from src.trainer.trainer import Trainer
from src.metrics.metrics import evaluate_model
from src.metrics.metrics import save_result
from src.metrics.metrics import Model_Storage
from src.visualization.plots import save_plots_same_model
import torch
import gc
from pathlib import Path


def main():

    #Setting paths
    batch_size256 = Path("experiments/configs/Van/batch_size256.yaml")
    batch_size512 = Path("experiments/configs/Van/batch_size512.yaml")
    batch_size1024 = Path("experiments/configs/Van/batch_size1024.yaml")

    learning_rate01 = Path("experiments/configs/Van/learning_rate01.yaml")
    learning_rate001 = Path("experiments/configs/Van/learning_rate001.yaml")
    learning_rate0001 = Path("experiments/configs/Van/learning_rate0001.yaml")

    num_layers1 = Path("experiments/configs/Van/num_layers1.yaml")
    num_layers2 = Path("experiments/configs/Van/num_layers2.yaml")
    num_layers4 = Path("experiments/configs/Van/num_layers4.yaml")

    path_table_learning_rate = [learning_rate01, learning_rate001, learning_rate0001]
    path_table_batch_size = [batch_size256, batch_size512, batch_size1024]
    path_table_num_layers = [num_layers1, num_layers2, num_layers4]

    hiperparameters_table = [path_table_batch_size, path_table_learning_rate, path_table_num_layers]
    hiperparameters_names = ["Batch size (van)", "Learning rate (van)", "Number of layers (van)"]
    
    model_names =[['batch_size256', 'batch_size512', 'batch_size1024'], 
                  ['learning_rate01', 'learning_rate001', 'learning_rate0001'], 
                  ['num_layers1', 'num_layers2', 'num_layers4']]

    """===VAN==="""
    i=0
    for path_table in hiperparameters_table:
        data_table=[]
        for path in path_table:
            print("---Initializing VAN model ---")

            config_van = load_config(path)
            model = Trainer(model=config_van['model'],
                      n=config_van['lattice_size'],
                      beta =config_van['beta'],
                      epochs=config_van['training']['epochs'],
                      lr=config_van['training']['learning_rate'],
                      batch_size=config_van['training']['batch_size'],
                      n_layers=config_van['model_params']['nb_layers'],
                      hiperparam_experiment = True)
            
            print("--- Training start ---")
            model.train_model()
            print("--- Training complete ---")
            
            print("--- Saving results ---")
            results = evaluate_model(model)
            save_result(results, "hiperparam_test", "VAN")

            data = Model_Storage(model)
            data_table.append(data)
            del results
            del model
            del config_van
            del data
            gc.collect()
            torch.cuda.empty_cache()

        save_plots_same_model(data_table, folder = "hiperparam_test", hiperparam_name=hiperparameters_names[i], model_names=model_names[i])
        i+=1

        del data_table

if __name__ == "__main__":
    main()