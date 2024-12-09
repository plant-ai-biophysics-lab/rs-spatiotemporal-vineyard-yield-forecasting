import torch
import argparse
import numpy as np


from src import DataLoader, engine
from model import UNet2DConvLSTM


def main(args):
    # Set random seed for reproducibility
    seed = 1987 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Check if there is GPU(s): {torch.cuda.is_available()}")

    # Extract arguments from args
    Exp_name = args.exp_name
    spatial_resolution = args.sr
    scenario = args.scenario
    batch_size = args.batch_size
    in_channels = args.in_channels
    num_filters = args.num_filters
    embd_channels = args.embd_channels
    dropout = args.dropout
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    resampling = args.resampling
    bottelneck_size = args.bottelneck_size

    # Load data
    data_loader_training, data_loader_validate, data_loader_test = DataLoader.dataloaders(
        spatial_resolution=spatial_resolution, 
        scenario=scenario,
        batch_size=batch_size, 
        in_channels=in_channels, 
        patch_size=16, 
        patch_offset=2, 
        cultivar_list=None,
        year_list=['2016', '2019', '2018', '2017'],
        resmapling_status=resampling,
        exp_name=Exp_name
    )

    
    # Initialize model
    model = UNet2DConvLSTM(
        in_channels=in_channels, 
        out_channels=1, 
        num_filters=num_filters, 
        embd_channels=embd_channels,
        dropout=dropout, 
        batch_size=batch_size, 
        bottelneck_size=bottelneck_size
    ).to(device)

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # Initialize training engine
    YE = engine.YieldEst(
        model, 
        lr=lr, 
        wd=wd, 
        exp=Exp_name
    )

    # Train the model
    # _ = YE.train(
    #     data_loader_training, 
    #     data_loader_validate, 
    #     epochs=epochs, 
    #     loss_stop_tolerance=100
    # )

    # Make predictions
    _ = YE.predict(model, data_loader_training, category='train', iter=1)
    _ = YE.predict(model, data_loader_validate, category='valid', iter=1)
    _ = YE.predict(model, data_loader_test, category='test', iter=10)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name", type=str, default = "test", help = "Experiment name")
    parser.add_argument("--sr", type=int, default = 10, help = "The spatial resolution of model training: 1 or 10")
    parser.add_argument("--scenario", type=str, default = "block-hold-out", help = "Recall the scenario: pixel-hold-out, year-hold-out, block-hold-out")
    parser.add_argument("--batch_size", type=int, default = 64, help = "Batch size")
    parser.add_argument("--in_channels", type=int, default = 6, help = "Number of input channels")
    parser.add_argument("--dropout", type=float, default = 0.3, help = "Amount of dropout")
    parser.add_argument("--lr", type=float, default = 0.001, help = "Learning rate")
    parser.add_argument("--wd", type=float, default = 0.05, help = "Value of weight decay")
    parser.add_argument("--epochs", type=int, default = 500, help = "The number of epochs")
    parser.add_argument("--num_filters", type=int, default = 16, help = "The number of initial filters, try 64")
    parser.add_argument("--embd_channels", type=int, default = 4, help = "The number of management inputs")
    parser.add_argument("--bottelneck_size", type=int, default = 2, help = "The size of image in bottelmeck for UNet, for 1m it should be 10 and for 10m it should be 2")
    parser.add_argument("--resampling",  type=str,   default = False, help = "Weight resampling status") 

    args = parser.parse_args()

    main(args)