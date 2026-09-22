from src.metrics.metrics import load_config
from src.models.transformer import Transformer_MODEL
from src.trainer.trainer import Trainer
from src.metrics.metrics import evaluate_model
from src.metrics.metrics import save_result
from pathlib import Path


def main():

    #Setting paths
    path_transf = Path("experiments/configs/transformer.yaml")
    path_lstm = Path("experiments/configs/lstm.yaml")
    path_pixelcnn = Path("experiments/configs/pixelcnn.yaml")
    path_van = Path("experiments/configs/van.yaml")

    #Loading config
    config_transf = load_config(path_transf)
    config_lstm = load_config(path_lstm)
    config_pixelcnn = load_config(path_pixelcnn)
    config_van = load_config(path_van)


    model = Transformer_MODEL

    #Setting trainers
    transformer = Trainer(model=config_transf['model'],
                      n=config_transf['lattice_size'],
                      beta =config_transf['beta'],
                      epochs=config_transf['training']['epochs'],
                      lr=config_transf['training']['learning_rate'],
                      batch_size=config_transf['training']['batch_size'],
                      n_layers=config_transf['model_params']['num_layers'],
                      embedding_dim=config_transf['model_params']['embedding_dim'],
                      num_heads=config_transf['model_params']['num_heads'])

    transformer.train_model()

    # lstm = Trainer(model=config_lstm['model'],
    #                   n=config_lstm['lattice_size'],
    #                   beta =config_lstm['beta'],
    #                   epochs=config_lstm['training']['epochs'],
    #                   lr=config_lstm['training']['learning_rate'],
    #                   batch_size=config_lstm['training']['batch_size'],
    #                   spin_placement_info=config_lstm['model_params']['spin_placement_info'])

    # lstm.train_model()

    # pixelcnn = Trainer(model=config_pixelcnn['model'],
    #                   n=config_pixelcnn['lattice_size'],
    #                   beta =config_pixelcnn['beta'],
    #                   epochs=config_pixelcnn['training']['epochs'],
    #                   lr=config_pixelcnn['training']['learning_rate'],
    #                   batch_size=config_pixelcnn['training']['batch_size'],
    #                   kernel_size=config_pixelcnn['model_params']['kernel_size'],
    #                   num_of_layers=config_pixelcnn['model_params']['num_layers'],
    #                   channels=config_pixelcnn['model_params']['channels'])

    # pixelcnn.train_model()

    # van = Trainer(model=config_van['model'],
    #                   n=config_van['lattice_size'],
    #                   beta =config_van['beta'],
    #                   epochs=config_van['training']['epochs'],
    #                   lr=config_van['training']['learning_rate'],
    #                   batch_size=config_van['training']['batch_size'],
    #                   nb_layers=config_van['model_params']['nb_layers'],
    #                   )

    # van.train_model()




    results = evaluate_model(transformer)
    save_result(results)

if __name__ == '__main__':
    main()