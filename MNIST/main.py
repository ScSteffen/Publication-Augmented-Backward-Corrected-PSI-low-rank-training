import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import src.low_rank_neural_networks as NetTrainer
import logging
from pathlib import Path
from datetime import datetime
import time 
import wandb
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Parse Arguments")

# Add arguments with default values
parser.add_argument('-batch_size', type=int, default=64, help="batch_size, default 64")
parser.add_argument('-lr', type=float, default=0.01, help="default 0.01")
parser.add_argument('-num_epochs', type=int, default=50, help="default 50")
parser.add_argument('-architecture', type=str, default="dense_architecture", help="dense_architecture, PSI_low_rank_architecture_4layer, ...")
parser.add_argument('-optimizer', type=str, default="SGD", help="SGD, Adam, default SGD")
parser.add_argument('-tol', type=float, default=0.01, help="tolerance e.g. 0.4,0.2,0.1,0.05,0.02,0.01, default 0.01")
parser.add_argument('-rank', type=int, default=20, help="default 20")
parser.add_argument('-wandb', type=int, default=0, help="default False")

# Parse the arguments
args = parser.parse_args()

# Define the hyperparameters
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.num_epochs
# "PSI_low_rank_architecture_4layer", ...
architecture = args.architecture
# "SGD", "Adam"
optimizer = args.optimizer
# tolerance e.g. 0.4, 0.2, 0.1, 0.05, 0.02, 0.01
tol = args.tol
# rank 
rank = args.rank

# define output path
now = datetime.now()
folder_name = now.strftime("%Y-%m-%d_%H-%M") + "_" + architecture + "New_epochs" + str(num_epochs) + "_lr" + str(learning_rate).split(".")[-1] + "_" + optimizer + "_tolerance_x100" + str(tol * 100)
output_path = Path.cwd() / Path('trained_models') / Path(folder_name)

# define logging info
watermark = "modelChoice{}_lr{}_batchSize{}_epochs{}_optimizer{}_initRank{}_tolerance{}".format(
                architecture,
                learning_rate,
                batch_size,
                num_epochs,
                optimizer,
                rank,
                tol,
            )

# define logger
logger_name = Path(__file__).parent / Path('trained_models')
logger_name = logger_name / Path(folder_name + '.log')

logging.basicConfig(filename = logger_name, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logging.info(watermark)

# init weights&biases for logging
if(args.wandb):
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Paper_run_bias_after_L{}".format(architecture),
        name=watermark,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "optimizer": optimizer,
            "init_rank": rank,
            "tol": tol,
        },
    )

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# define architectures of dense and dlrt networks
if(architecture == 'dense_architecture'):
    integrator = 'Dense'
    architecture_model = [
    {'type': 'dense', 'dims': [784, 500]},
    {'type': 'dense', 'dims': [500, 500]},
    {'type': 'dense', 'dims': [500, 10]}
]

if(architecture == 'dense_architecture_4layer'):
    integrator = 'Dense'
    architecture_model = [
    {'type': 'dense', 'dims': [784, 500]},
    {'type': 'dense', 'dims': [500, 500]},
    {'type': 'dense', 'dims': [500, 500]},
    {'type': 'dense', 'dims': [500, 500]},
    {'type': 'dense', 'dims': [500, 10]}
]

if(architecture == 'PSI_low_rank_architecture'):
    integrator = 'PSI'
    architecture_model = [
    {'type': 'PSI_dynamical_low_rank', 'dims': [784, 500], 'rank': rank},
    {'type': 'PSI_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'dense', 'dims': [500, 10]}
    ]

if(architecture == 'PSI_low_rank_architecture_4layer'):
    integrator = 'PSI'
    architecture_model = [
    {'type': 'PSI_dynamical_low_rank', 'dims': [784, 500], 'rank': rank},
    {'type': 'PSI_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'PSI_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'PSI_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'dense', 'dims': [500, 10]}
    ]

if(architecture == 'PSI_Backward_low_rank_architecture'):
    integrator = 'PSI_Backward'
    architecture_model = [
    {'type': 'PSI_Backward_dynamical_low_rank', 'dims': [784, 500], 'rank': rank},
    {'type': 'PSI_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'dense', 'dims': [500, 10]}
    ]

if(architecture == 'PSI_Backward_low_rank_architecture_4layer'):
    integrator = 'PSI_Backward'
    architecture_model = [
    {'type': 'PSI_Backward_dynamical_low_rank', 'dims': [784, 500], 'rank': rank},
    {'type': 'PSI_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'PSI_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'PSI_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank},
    {'type': 'dense', 'dims': [500, 10]}
    ]
      
if(architecture == 'PSI_Augmented_Backward_low_rank_architecture'):
    integrator = 'PSI_Augmented_Backward'
    architecture_model = [
        {'type': 'PSI_Augmented_Backward_dynamical_low_rank', 'dims': [784, 500], 'rank': rank, 'tol': tol},
        {'type': 'PSI_Augmented_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank, 'tol': tol},
        {'type': 'dense', 'dims': [500, 10]}
    ]

if(architecture == 'PSI_Augmented_Backward_low_rank_architecture_4layer'):
    integrator = 'PSI_Augmented_Backward'
    architecture_model = [
    {'type': 'PSI_Augmented_Backward_dynamical_low_rank', 'dims': [784, 500], 'rank': rank, 'tol': tol},
    {'type': 'PSI_Augmented_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank, 'tol': tol},
    {'type': 'PSI_Augmented_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank, 'tol': tol},
    {'type': 'PSI_Augmented_Backward_dynamical_low_rank', 'dims': [500, 500], 'rank': rank, 'tol': tol},
    {'type': 'dense', 'dims': [500, 10]}
    ]

t = NetTrainer.Trainer(architecture_model, train_loader, test_loader, integrator)
t.train(num_epochs, learning_rate, integrator, args)
t.model.write(output_path)
print("Training finished.")
