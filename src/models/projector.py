import torch


class HeteroProjector(torch.nn.Module):
    """
    Project node-type specific features into a shared latent space.
    """
    def __init__(self, input_dims, hidden_dim):
        super(HeteroProjector, self).__init__()
        self.hidden_dim = hidden_dim
        self.projectors = torch.nn.ModuleDict({
            node_type: torch.nn.Linear(input_dim, hidden_dim)
            for node_type, input_dim in input_dims.items()
        })
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.projectors.values():
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x_dict):
        return {
            node_type: self.projectors[node_type](features.float())
            for node_type, features in x_dict.items()
        }

    @torch.no_grad()
    def project(self, x_dict):
        self.eval()
        return self.forward(x_dict)

    @torch.no_grad()
    def project_and_unify(self, x_dict, node_types):
        projected_x_dict = self.project(x_dict)
        unified_embeddings = torch.cat([projected_x_dict[node_type] for node_type in node_types], dim=0)
        return projected_x_dict, unified_embeddings
