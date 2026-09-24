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
    batch_size256 = Path("experiments/configs/Transformers/batch_size256.yaml")
    batch_size512 = Path("experiments/configs/Transformers/batch_size512.yaml")
    batch_size1024 = Path("experiments/configs/Transformers/batch_size1024.yaml")

    emb_dim32 = Path("experiments/configs/Transformers/emb_dim32.yaml")
    emb_dim64 = Path("experiments/configs/Transformers/emb_dim64.yaml")
    emb_dim48 = Path("experiments/configs/Transformers/emb_dim48.yaml")

    num_heads2 = Path("experiments/configs/Transformers/num_heads2.yaml")
    num_heads4 = Path("experiments/configs/Transformers/num_heads4.yaml")
    num_heads8 = Path("experiments/configs/Transformers/num_heads8.yaml")

    path_table_num_heads = [num_heads2, num_heads4, num_heads8]
    path_table_batch_size = [batch_size256, batch_size512, batch_size1024]
    path_table_emd_dim = [emb_dim32, emb_dim64, emb_dim48]

    hiperparameters_table = [path_table_batch_size, path_table_emd_dim, path_table_num_heads]
    hiperparameters_names = ["Batch size (transfrm)", "Embedding dimensions (transfrm)", "Number of Heads (transfrm)"]
    
    model_names =[['batch_size256', 'batch_size512', 'batch_size1024'], ['emb_dim32', 'emb_dim64', 'emb_dim48'], ['num_heads2', 'num_heads4', 'num_heads8']]

    """===TRANSFORMER==="""
    i=0
    for path_table in hiperparameters_table:
        data_table=[]
        for path in path_table:
            print("---Initializing TRANSFORMER model ---")

            config_transf = load_config(path)
            model = Trainer(model=config_transf['model'],
                            n=config_transf['lattice_size'],
                            beta =config_transf['beta'],
                            epochs=config_transf['training']['epochs'],
                            lr=config_transf['training']['learning_rate'],
                            batch_size=config_transf['training']['batch_size'],
                            n_layers=config_transf['model_params']['num_layers'],
                            hiperparam_experiment = True,
                            embedding_dim=config_transf['model_params']['embedding_dim'],
                            num_heads=config_transf['model_params']['num_heads'])
            
            print("--- Training start ---")
            model.train_model()
            print("--- Training complete ---")
            
            print("--- Saving results ---")
            results = evaluate_model(model)
            save_result(results, "hiperparam_test", "Transformer")

            data = Model_Storage(model)
            data_table.append(data)
            del results
            del model
            del config_transf
            del data
            gc.collect()
            torch.cuda.empty_cache()

        save_plots_same_model(data_table, folder = "hiperparam_test", hiperparam_name=hiperparameters_names[i], model_names=model_names[i])
        i+=1

        del data_table

if __name__ == "__main__":
    main()