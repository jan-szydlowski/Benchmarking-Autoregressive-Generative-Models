from src.metrics.metrics import load_config
from src.trainer.trainer import Trainer
from src.metrics.metrics import evaluate_model
from src.metrics.metrics import save_result
from src.metrics.metrics import Model_Storage
from experiments.hyperparams.run_lstm import run_lstm
from experiments.hyperparams.run_van import run_van
from experiments.hyperparams.run_transformer import run_transformer
from experiments.hyperparams.run_pixelcnn import run_pixelcnn
from src.visualization.plots import save_plots_diff_model
import sys
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

def main():
    set_seed(7)
    try:
        test_kind = sys.argv[0]
    except:
        print("Missing first argument")

    if test_kind == 'a' or test_kind == 'e':
        #Setting paths
        path_transf = Path("experiments/configs/best_models/transformer.yaml")
        path_lstm = Path("experiments/configs/best_models/lstm.yaml")
        path_pixelcnn = Path("experiments/configs/best_models/pixelcnn.yaml")
        path_van = Path("experiments/configs/best_models/van.yaml")


        """===TRANSFORMER==="""

        print("---Initializing TRANSFORMER model ---")

        config_transf = load_config(path_transf)
        transformer = Trainer(model=config_transf['model'],
                        n=config_transf['lattice_size'],
                        beta =config_transf['beta'],
                        epochs=config_transf['training']['epochs'],
                        lr=config_transf['training']['learning_rate'],
                        batch_size=config_transf['training']['batch_size'],
                        n_layers=config_transf['model_params']['num_layers'],
                        embedding_dim=config_transf['model_params']['embedding_dim'],
                        num_heads=config_transf['model_params']['num_heads'])
        
        print("--- Training start ---")
        transformer.train_model()
        print("--- Training complete ---")
        
        print("--- Saving results ---")
        results = evaluate_model(transformer)
        save_result(results, "final_models", "comparison_table")

        transformer_data = Model_Storage(transformer)

        del results
        del transformer
        del config_transf

        gc.collect()
        torch.cuda.empty_cache()

        """===LSTM==="""

        print("---Initializing LSTM model ---")

        config_lstm = load_config(path_lstm)
        lstm = Trainer(model=config_lstm['model'],
                        n=config_lstm['lattice_size'],
                        beta =config_lstm['beta'],
                        epochs=config_lstm['training']['epochs'],
                        lr=config_lstm['training']['learning_rate'],
                        batch_size=config_lstm['training']['batch_size'],
                        spin_placement_info=config_lstm['model_params']['spin_placement_info'],
                        n_layers=1)
        
        print("--- Training start ---")
        lstm.train_model()
        print("--- Training complete ---")
        
        print("--- Saving results ---")
        results = evaluate_model(lstm)
        save_result(results, "final_models", "comparison_table")

        lstm_data = Model_Storage(lstm)

        del results
        del lstm
        del config_lstm

        gc.collect()
        torch.cuda.empty_cache()


        """===PIXELCNN==="""

        print("---Initializing PIXELCNN model ---")

        config_pixelcnn = load_config(path_pixelcnn)
        pixelcnn = Trainer(model=config_pixelcnn['model'],
                        n=config_pixelcnn['lattice_size'],
                        beta =config_pixelcnn['beta'],
                        epochs=config_pixelcnn['training']['epochs'],
                        lr=config_pixelcnn['training']['learning_rate'],
                        batch_size=config_pixelcnn['training']['batch_size'],
                        kernel_size=config_pixelcnn['model_params']['kernel_size'],
                        n_layers=config_pixelcnn['model_params']['num_layers'],
                        channels=config_pixelcnn['model_params']['channels'])

        print("--- Training start ---")
        pixelcnn.train_model()
        print("--- Training complete ---")

        print("--- Saving results ---")
        results = evaluate_model(pixelcnn)
        print("--- Creating plots ---")
        save_result(results, "final_models", "comparison_table")

        pixelcnn_data = Model_Storage(pixelcnn)

        del results
        del pixelcnn
        del config_pixelcnn

        gc.collect()
        torch.cuda.empty_cache()


        """===VAN==="""

        print("---Initializing VAN model ---")

        config_van = load_config(path_van)   
        van = Trainer(model=config_van['model'],
                        n=config_van['lattice_size'],
                        beta =config_van['beta'],
                        epochs=config_van['training']['epochs'],
                        lr=config_van['training']['learning_rate'],
                        batch_size=config_van['training']['batch_size'],
                        n_layers=config_van['model_params']['nb_layers'])
        
        print("--- Training start ---")
        van.train_model()
        print("--- Training complete ---")
        
        print("--- Saving results ---")
        results = evaluate_model(van)
        save_result(results, "final_models", "comparison_table")

        van_data = Model_Storage(van)

        del results
        del van
        del config_van

        gc.collect()
        torch.cuda.empty_cache()

    save_plots_diff_model(lstm_data, van_data, transformer_data, pixelcnn_data, folder = "final_models")
    if test_kind == 'a' or test_kind == 'h':
        print("===Starting Hiperparam Test===")
        run_pixelcnn()
        run_van()
        run_transformer()
        run_lstm()


if __name__ == '__main__':
    main()