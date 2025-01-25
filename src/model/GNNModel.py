
import torch
import torch.nn as nn

from torch_geometric.nn import GATConv


class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, expected_n_points=None, expected_scale=None):
        super(GNNModel, self).__init__()
        
        self.conv1_1 = GATConv(input_size, hidden_size, heads=num_heads, concat=True)
        self.conv2_1 = GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, concat=False)
        self.fc_1 = nn.Linear(hidden_size, output_size)

        self.conv1_2 = GATConv(input_size, hidden_size, heads=num_heads, concat=True)
        self.conv2_2 = GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, concat=False)
        self.fc_2 = nn.Linear(hidden_size, output_size)

        self.conv1_3 = GATConv(input_size, hidden_size, heads=num_heads, concat=True)
        self.conv2_3 = GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, concat=False)
        self.fc_3 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()

        if expected_n_points and expected_scale:
            self.graphs = self.get_graphs(expected_n_points, expected_scale)

    def forward(self, data): 
        if 'edge_index_1' not in data.keys():
            edge_index_1, edge_index_2, edge_index_3 = self.graphs
        else:
            edge_index_1, edge_index_2, edge_index_3 = data.edge_index_1, data.edge_index_2, data.edge_index_3

        x1 = self.conv1_1(data.x, edge_index_1)
        x1 = self.conv2_1(x1, edge_index_1)
        x1 = self.fc_1(x1)

        x2 = self.conv1_2(data.x, edge_index_2)
        x2 = self.conv2_2(x2, edge_index_2)
        x2 = self.fc_2(x2)

        x3 = self.conv1_3(data.x, edge_index_3)
        x3 = self.conv2_3(x3, edge_index_3)
        x3 = self.fc_3(x3)

        if self.training:
            return x1, x2, x3
        else:
            x = data.x[:, 1].unsqueeze(-1)

            curves = torch.cat([x, x1, x2, x3], dim=1)
            pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

            mean_curves = (curves[:, pairs[:, 0]] + curves[:, pairs[:, 1]]) / 2
            diff_curves = torch.abs(curves[:, pairs[:, 0]] - curves[:, pairs[:, 1]])
            indices = torch.min(diff_curves, dim=1).indices

            result = torch.gather(mean_curves, 1, indices.unsqueeze(1))
            return result
        
    def get_graphs(self, n_points, scale=1):
        edge_index_1 = torch.tensor(
            [[i - 1 * scale, i] for i in range(1 * scale, n_points)] + 
            [[i - 3 * scale, i] for i in range(3 * scale, n_points)] +
            [[i - 5 * scale, i] for i in range(5 * scale, n_points)] +
            [[i - 7 * scale, i] for i in range(7 * scale, n_points)] +
            [[i - 9 * scale, i] for i in range(9 * scale, n_points)] +
            [[i - 11 * scale, i] for i in range(11 * scale, n_points)],
            dtype=torch.long
        ).t().contiguous()

        edge_index_2 = torch.tensor(
            [[i + 1 * scale, i] for i in range(n_points - 1 * scale)] +
            [[i + 3 * scale, i] for i in range(n_points - 3 * scale)] +
            [[i + 5 * scale, i] for i in range(n_points - 5 * scale)] +
            [[i + 7 * scale, i] for i in range(n_points - 7 * scale)] +
            [[i + 9 * scale, i] for i in range(n_points - 9 * scale)] +
            [[i + 11 * scale, i] for i in range(n_points - 11 * scale)],
            dtype=torch.long
        ).t().contiguous()

        edge_index_3 = torch.cat((edge_index_1, edge_index_2), dim=1)

        return edge_index_1, edge_index_2, edge_index_3

    