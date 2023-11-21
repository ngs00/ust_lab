import torch
from torch.nn.functional import leaky_relu
from torch_geometric.nn.conv import *
from torch_geometric.nn.glob import *


class GCN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_out):
        super(GCN, self).__init__()
        self.nfc = torch.nn.Linear(dim_node_feat, 128)
        self.gc1 = GCNConv(128, 128)
        self.gc2 = GCNConv(128, 128)
        self.fc1 = torch.nn.Linear(128, 32)
        self.fc2 = torch.nn.Linear(32, dim_out)

        self.nfc.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, g):
        hx = leaky_relu(self.nfc(g.x))
        hx = leaky_relu(self.gc1(hx, g.edge_index))
        hx = leaky_relu(self.gc2(hx, g.edge_index))
        hg = global_mean_pool(hx, g.batch)
        hg = leaky_relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


class GAT(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_out):
        super(GAT, self).__init__()
        self.nfc = torch.nn.Linear(dim_node_feat, 128)
        self.gc1 = GATv2Conv(128, 128)
        self.gc2 = GATv2Conv(128, 128)
        self.fc1 = torch.nn.Linear(128, 32)
        self.fc2 = torch.nn.Linear(32, dim_out)

        self.nfc.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, g):
        hx = leaky_relu(self.nfc(g.x))
        hx = leaky_relu(self.gc1(hx, g.edge_index))
        hx = leaky_relu(self.gc2(hx, g.edge_index))
        hg = global_mean_pool(hx, g.batch)
        hg = leaky_relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


class GCN_ADD(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_out):
        super(GCN_ADD, self).__init__()
        self.nfc = torch.nn.Linear(dim_node_feat, 128)
        self.gc1 = GCNConv(128, 128)
        self.gc2 = GCNConv(128, 128)
        self.fc1 = torch.nn.Linear(128, 32)
        self.fc2 = torch.nn.Linear(32, dim_out)

        self.nfc.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, g):
        hx = leaky_relu(self.nfc(g.x))
        hx = leaky_relu(self.gc1(hx, g.edge_index))
        hx = leaky_relu(self.gc2(hx, g.edge_index))
        hg = global_add_pool(hx, g.batch)
        hg = leaky_relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


class MPNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_out):
        super(MPNN, self).__init__()
        self.nfc = torch.nn.Linear(dim_node_feat, 64)
        self.efc1 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 64 * 64))
        self.gc1 = NNConv(64, 64, self.efc1)
        self.efc2 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 64 * 64))
        self.gc2 = NNConv(64, 64, self.efc2)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, dim_out)

        self.nfc.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, g):
        hx = leaky_relu(self.nfc(g.x))
        hx = leaky_relu(self.gc1(hx, g.edge_index, g.edge_attr))
        hx = leaky_relu(self.gc2(hx, g.edge_index, g.edge_attr))
        hg = global_add_pool(hx, g.batch)
        hg = leaky_relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


class CGCNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_out):
        super(CGCNN, self).__init__()
        self.nfc = torch.nn.Linear(dim_node_feat, 128)
        self.gc1 = CGConv(128, dim_edge_feat, batch_norm=True)
        self.gc2 = CGConv(128, dim_edge_feat, batch_norm=True)
        self.enfc = torch.nn.Linear(128, 32)
        self.fc1 = torch.nn.Linear(32, dim_out)

        self.nfc.reset_parameters()
        self.enfc.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, g):
        hx = leaky_relu(self.nfc(g.x))
        hx = leaky_relu(self.gc1(hx, g.edge_index, g.edge_attr))
        hx = leaky_relu(self.gc2(hx, g.edge_index, g.edge_attr))
        z = normalize(global_mean_pool(hx, g.batch), p=2, dim=1)
        z = leaky_relu(self.enfc(z))
        out = self.fc1(z)

        return out


def fit_gnn(model, data_loader, optimizer, loss_func):
    train_loss = 0

    model.train()
    for b in data_loader:
        b = b.cuda()
        preds = model(b)
        loss = loss_func(b.y, preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def predict_gnn(model, data_loader):
    model.eval()
    with torch.no_grad():
        return torch.vstack([model(b.cuda()) for b in data_loader]).cpu()
