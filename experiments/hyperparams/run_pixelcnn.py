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
    batch_size128 = Path("experiments/configs/Pixelcnn/batch_size128.yaml")
    batch_size256 = Path("experiments/configs/Pixelcnn/batch_size256.yaml")
    batch_size516 = Path("experiments/configs/Pixelcnn/batch_size516.yaml")

    channels16 = Path("experiments/configs/Pixelcnn/channels16.yaml")
    channels32 = Path("experiments/configs/Pixelcnn/channels32.yaml")
    channels64 = Path("experiments/configs/Pixelcnn/channels64.yaml")

    kernel_size3 = Path("experiments/configs/Pixelcnn/kernel_size3.yaml")
    kernel_size5 = Path("experiments/configs/Pixelcnn/kernel_size5.yaml")
    kernel_size7 = Path("experiments/configs/Pixelcnn/kernel_size7.yaml")

    path_table_channels = [channels16, channels32, channels64]
    path_table_batch_size = [batch_size128, batch_size256, batch_size516]
    path_table_kernel_size = [kernel_size3, kernel_size5, kernel_size7]

    hiperparameters_table = [path_table_batch_size, path_table_channels, path_table_kernel_size]
    hiperparameters_names = ["Batch size (pixelcnn)", "Number of channels (pixelcnn)", "Kernel size (pixelcnn)"]
    
    model_names =[['batch_size128', 'batch_size256', 'batch_size516'], 
                  ['channels16', 'channels32', 'channels64'], 
                  ['kernel_size3', 'kernel_size5', 'kernel_size7']]

    """===PIXELCNN==="""
    i=0
    for path_table in hiperparameters_table:
        data_table=[]
        for path in path_table:
            print("---Initializing PIXELCNN model ---")

            config_pixelcnn = load_config(path)
            model = Trainer(model=config_pixelcnn['model'],
                      n=config_pixelcnn['lattice_size'],
                      beta =config_pixelcnn['beta'],
                      epochs=config_pixelcnn['training']['epochs'],
                      lr=config_pixelcnn['training']['learning_rate'],
                      batch_size=config_pixelcnn['training']['batch_size'],
                      kernel_size=config_pixelcnn['model_params']['kernel_size'],
                      n_layers=config_pixelcnn['model_params']['num_layers'],
                      hiperparam_experiment = True,
                      channels=config_pixelcnn['model_params']['channels'])
            
            print("--- Training start ---")
            model.train_model()
            print("--- Training complete ---")
            
            print("--- Saving results ---")
            results = evaluate_model(model)
            save_result(results, "hiperparam_test", "PixelCNN")

            data = Model_Storage(model)
            data_table.append(data)
            del results
            del model
            del config_pixelcnn
            del data
            gc.collect()
            torch.cuda.empty_cache()

        save_plots_same_model(data_table, folder = "hiperparam_test", hiperparam_name=hiperparameters_names[i], model_names=model_names[i])
        i+=1

        del data_table

if __name__ == "__main__":
    main()