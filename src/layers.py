import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import SAGEConv

class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, args, input_channels, output_channels):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features. 
        """
        super(StackedGCN, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers = []
        self._layer_sizes = [self.input_channels] + list(self.args.layers) + [self.output_channels]
        for i, _ in enumerate(self._layer_sizes[:-1]):
            self.layers.append(GCNConv(self._layer_sizes[i], self._layer_sizes[i+1]))
        self.layers = ListModule(*self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            features = torch.nn.functional.relu(self.layers[i](features, edges))
            if i > 1:
                features = torch.nn.functional.dropout(features, p = self.args.dropout, training = self.training)
        features = self.layers[num_layers - 1](features, edges)
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions


class HeteroStackedGNN(torch.nn.Module):
    """
    Multi-layer heterogeneous GNN with target-node classification.
    """
    def __init__(self, args, metadata, input_channels, output_channels, target_node_type):
        super(HeteroStackedGNN, self).__init__()
        self.args = args
        self.metadata = metadata
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_node_type = target_node_type
        self.setup_layers()

    def setup_layers(self):
        self.layers = torch.nn.ModuleList()
        hidden_layers = list(self.args.layers)
        if not hidden_layers:
            hidden_layers = [self.input_channels]
        layer_sizes = [self.input_channels] + hidden_layers
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            relation_layers = {
                edge_type: SAGEConv((input_size, input_size), output_size)
                for edge_type in self.metadata[1]
            }
            self.layers.append(HeteroConv(relation_layers, aggr="sum"))
        self.output_layer = torch.nn.Linear(layer_sizes[-1], self.output_channels)

    def forward(self, x_dict, edge_index_dict):
        for layer_index, layer in enumerate(self.layers):
            updated_x_dict = layer(x_dict, edge_index_dict)
            x_dict = {
                node_type: updated_x_dict.get(node_type, features)
                for node_type, features in x_dict.items()
            }
            if layer_index < len(self.layers) - 1:
                x_dict = {
                    node_type: torch.nn.functional.dropout(
                        torch.nn.functional.relu(features),
                        p=self.args.dropout,
                        training=self.training
                    )
                    for node_type, features in x_dict.items()
                }
        logits = self.output_layer(x_dict[self.target_node_type])
        return torch.nn.functional.log_softmax(logits, dim=1)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
