import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGTMultiLabel(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for multi-label therapeutic strategy classification.
    Processes 'message', 'conversation', 'lexicon', and 'distress' nodes.
    """

    def __init__(
        self,
        metadata,
        in_channels_dict,
        hidden_dim=256,
        out_dim=3,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        lexicon_x=None,
        distress_x=None
    ):
        super().__init__()

        self.metadata = metadata
        self.dropout = nn.Dropout(dropout)

        self.lin_dict = nn.ModuleDict({
            ntype: Linear(in_channels_dict[ntype], hidden_dim)
            for ntype in metadata[0]
            if ntype not in ["lexicon", "distress"]
        })

        if lexicon_x is not None:
            self.lexicon_emb = nn.Parameter(lexicon_x.clone().detach())
            lexicon_dim = lexicon_x.size(1)
        else:
            self.lexicon_emb = None
            lexicon_dim = in_channels_dict["lexicon"]

        if distress_x is not None:
            self.distress_emb = nn.Parameter(distress_x.clone().detach())
            distress_dim = distress_x.size(1)
        else:
            self.distress_emb = None
            distress_dim = in_channels_dict["distress"]

        self.lexicon_proj = Linear(lexicon_dim, hidden_dim)
        self.distress_proj = Linear(distress_dim, hidden_dim)

        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads
            )
            for _ in range(num_layers)
        ])

        self.out_lin = Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        h = {}

        for ntype, x in x_dict.items():
            if ntype == "lexicon":
                lex_x = self.lexicon_emb if self.lexicon_emb is not None else x
                h["lexicon"] = F.relu(self.lexicon_proj(lex_x))

            elif ntype == "distress":
                distress_x = self.distress_emb if self.distress_emb is not None else x
                h["distress"] = F.relu(self.distress_proj(distress_x))

            else:
                h[ntype] = F.relu(self.lin_dict[ntype](x))

        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {
                ntype: self.dropout(F.relu(node_h))
                for ntype, node_h in h.items()
            }

        logits = self.out_lin(h["message"])
        return logits, h