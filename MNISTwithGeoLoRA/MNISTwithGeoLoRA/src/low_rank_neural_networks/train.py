import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Network
import logging
import time 
import wandb
import numpy as np

class Trainer:
    def __init__(self, net_architecture, train_loader, test_loader, integrator):
        """Constructs trainer which manages and trains neural network
        Args:
            net_architecture: Dictionary of the network architecture. Needs keys 'type' and 'dims'. Low-rank layers need key 'rank'.
            train_loader: loader for training data
            test_loader: loader for test data
        """
        # Set the device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        # Initialize the model
        self.model = Network(net_architecture).to(self.device)

        # find all ids of dynamical low-rank layers, since these layer require two steps
        if(integrator == "PSI"):
            self.dlr_layer_ids = [index for index, layer in enumerate(net_architecture) if layer['type'] == 'PSI_dynamical_low_rank']
        if(integrator == "PSI_Backward"):
            self.dlr_layer_ids = [index for index, layer in enumerate(net_architecture) if layer['type'] == 'PSI_Backward_dynamical_low_rank']
        if(integrator == "PSI_Augmented_Backward"):
            self.dlr_layer_ids = [index for index, layer in enumerate(net_architecture) if layer['type'] == 'PSI_Augmented_Backward_dynamical_low_rank']
        if(integrator == "GeoLoRA"):
            self.dlr_layer_ids = [index for index, layer in enumerate(net_architecture) if layer['type'] == 'GeoLoRA']
        
        # store train and test data
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.best_acc = 0

    def train(self, num_epochs, learning_rate, integrator, args):
        """Trains neural network for specified number of epochs with specified learning rate
        Args:
            num_epochs: number of epochs for training
            learning_rate: learning rate for optimization method
            optimizer_type: used optimizer. Use Adam for vanilla training.
        """
        optimizer = args.optimizer

        # Define the loss function and optimizer. Optimizer is only needed to set all gradients to zero.
        criterion = nn.CrossEntropyLoss()
        if(optimizer == "SGD"): 
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif(optimizer == "Adam"): 
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else: 
            print("invalid opimizer chosen")
        
        if(integrator == "Dense"):
            steps = ['dense']
        if(integrator == "PSI"):
            steps = ['K', 'S', 'L']
        if(integrator == "PSI_Backward" or integrator == "PSI_Augmented_Backward"):
            steps = ['K', 'L']
        if(integrator == "GeoLoRA"):
            steps = ['KSL']

        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # checks only if dlr_layer_ids exists?
                for mode in steps:
                    # Forward pass
                    outputs = self.model.forward(data, mode)
                    loss = criterion(outputs, targets)
                    
                    # Backward to calculate gradients of coefficients
                    optimizer.zero_grad()
                    loss.backward()    

                    self.model.step(learning_rate, mode)
                                      
                # print progress
                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
                    logging.info(f"Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx+1}/{len(self.train_loader)}] Loss(Training): {loss.item():.4f} %")

            epoch_end_time = time.time()
            training_loss = float(loss.item())
            # evaluate model on test date
            if(integrator == "Dense"):
                if(args.architecture[-12:] == 'architecture'):
                    ranks_all_layers = ['nra', 'nra']
                else:
                    ranks_all_layers = ['nra', 'nra', 'nra', 'nra']
                accuracy, average_validation_loss = self.test_model(epoch, num_epochs)
            else:
                ranks_all_layers = self.model.get_ranks()
                accuracy, average_validation_loss = self.test_model(epoch, num_epochs, ranks_all_layers)

            best_acc = self.best_acc
            if(args.architecture[-12:] == 'architecture'):
                if(args.wandb):
                    wandb.log(
                        {
                        "loss train": training_loss,
                        "average_validation_loss": average_validation_loss,
                        "val_accuracy": accuracy,
                        "best val acc": best_acc,
                        "epoch": epoch+1,
                        "rank1 ": ranks_all_layers[0],
                        "rank2 ": ranks_all_layers[1],
                        "time elapsed": epoch_end_time - epoch_start_time,
                        },
                    )
            else:
                if(args.wandb):
                    wandb.log(
                        {
                        "loss train": training_loss,
                        "average_validation_loss": average_validation_loss,
                        "val_accuracy": accuracy,
                        "best val acc": best_acc,
                        "epoch": epoch+1,
                        "rank1 ": ranks_all_layers[0],
                        "rank2 ": ranks_all_layers[1],
                        "rank3 ": ranks_all_layers[2],
                        "rank4 ": ranks_all_layers[3],
                        "time elapsed": epoch_end_time - epoch_start_time,
                        },
                    )

            # stop training if gradient explodes
            if((epoch == 5) and (best_acc < 20)):
                break

    def test_model(self, epoch, num_epochs, ranks_all_layers = 'nra'):
        """Prints the model's accuracy on the test data
        """
        # Test the model
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            validation_loss = 0.0
            for data, targets in self.test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data, 'test')
                loss = criterion(outputs, targets)  # Calculate loss
                validation_loss += loss.item()  # Accumulate validation loss
            
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            accuracy = 100 * correct / total
            if(self.best_acc < accuracy): self.best_acc = accuracy
            print("num correct:", correct, "Rank:", ranks_all_layers, "num total:", total)
            print(f"Accuracy of the network on the test images: {accuracy}%")
            logging.info(f"Epoch [{epoch+1}/{num_epochs}] Rank: {ranks_all_layers} Loss(Training):  Accuracy: {accuracy}%")
        
            # Compute average validation loss
            average_validation_loss = validation_loss / len(self.test_loader)
        
        return accuracy, average_validation_loss
