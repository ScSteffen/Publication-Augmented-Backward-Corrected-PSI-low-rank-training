import torch
import torch.nn as nn
import src.low_rank_neural_networks.layers as layers


# Define custom neural network architecture from input dictionary
class Network(nn.Module):
    def __init__(self, net_architecture):
        """Constructs a neural network given its network architecture.
        Args:
            net_architecture: Dictionary of the network architecture. Needs keys 'type' and 'dims'. Low-rank layers need key 'rank'.
        """
        # define Network as child of nn.Module
        super(Network, self).__init__()
        self.layers = torch.nn.Sequential()

        # define intermediate layers
        for i, layer in enumerate(net_architecture[:len(net_architecture)-1]):
            self.layers.add_module(name=f"hidden_{i+1}", module=layers.create_layer(layer))

        # define output layer
        self.out = layers.create_layer(net_architecture[-1])

    def forward(self, x, mode):
        """Returns the output of the neural network. The formula implemented is z_k = ReLU(layer_k(z_{k-1})), where z_0 = x.
        Args:
            x: input image or batch of input images
        Returns:
            output neural network for given input
        """
        x = x.view(-1, 784)  # Flatten the input image
        # for all dlra layers (see __int__)
        for layer in self.layers:
            x = torch.relu(layer.forward(x, mode))
        # for the output layer (see __int__)
        x = self.out.forward(x)
        return x

    def step(self, learning_rate, mode):
        """Performs training step on all layers
        Args:
            learning_rate: learning rate for training
        Returns:
            output neural network for given input
        """
        if(mode == 'K'):
            self.out.step(learning_rate)

        for layer in self.layers:
            layer.step(learning_rate, mode)

    
    def set_all_zero(self):
        for layer in self.layers:
            layer.set_all_zero()
        self.out.set_all_zero()

    def get_ranks(self):
        rank_arr = []
        for layer in self.layers:
            rank_arr.append(layer.get_layer_rank())
        return rank_arr
        
    def write(self, output_path):
        """Writes weight matrices for all layers
        Args:
            folder_name: name of the folder in which weights are stored
        """  
        if not output_path.is_dir():
            output_path.mkdir()

        for name, layer in self.layers.named_children():
            layer.write(str(output_path.resolve() / name))

        self.out.write(str(output_path.resolve() / "output_layer"))
        