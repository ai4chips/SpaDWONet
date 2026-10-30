import torch
from torch import nn
import LNO.lno as lno
import Transolver.transolver as transolver
import ONO.ono as ono

NAMES = ("LNO", "Transolver", "ONO")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Operator2D(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        axis = torch.linspace(0, 1, 64)
        grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(1, 4096, 2)
        self.register_buffer("coordinates", grid, persistent=False)
        args = dict(config)
        ff_hidden = args.pop("ff_hidden", None)
        if name == "LNO":
            self.network = lno.LNO(**args)
            if ff_hidden is not None:
                for block in self.network.attn_blocks:
                    w = block.n_dim
                    block.mlp = nn.Sequential(nn.Linear(w, ff_hidden), nn.GELU(), nn.Linear(ff_hidden, w))
        elif name == "Transolver":
            self.network = transolver.Model(**args)
            
            
            self.network.placeholder.requires_grad_(False)
            if ff_hidden is not None:
                for block in self.network.blocks:
                    w = block.mlp.n_input
                    block.mlp = transolver.MLP(w, ff_hidden, w, n_layers=0, res=False, act="gelu")
                    self.network._init_weights(block.mlp.linear_pre[0])
                    self.network._init_weights(block.mlp.linear_post)
        elif name == "ONO":
            self.network = ono.ONO2(**args)
            if ff_hidden is not None:
                for block in self.network.blocks:
                    for attr in ("mlp", "mlp2"):
                        old = getattr(block, attr)
                        if isinstance(old, ono.MLP):
                            w = old.n_input
                            replacement = ono.MLP(w, ff_hidden, w, n_layers=0, res=False, act="gelu")
                            replacement.apply(self.network._init_weights)
                            setattr(block, attr, replacement)
        else:
            raise ValueError(name)

    def forward(self, power):
        if power.ndim != 4 or power.shape[1:] != (1, 64, 64):
            raise ValueError(f"Expected Bx1x64x64 input, received {tuple(power.shape)}")
        values = power.flatten(2).transpose(1, 2)
        xy = self.coordinates.expand(power.shape[0], -1, -1)
        prediction = self.network(xy, values)
        return prediction.transpose(1, 2).reshape(-1, 1, 64, 64)


BASE_CONFIGS = {
    "LNO": dict(n_block=4, n_mode=256, n_dim=128, n_head=8, n_layer=2,
                x_dim=2, y1_dim=1, y2_dim=1, attn="Attention_Vanilla", act="GELU", model_attr={"time": False}),
    "Transolver": dict(space_dim=2, n_layers=8, n_hidden=128, n_head=8, mlp_ratio=1,
                       fun_dim=1, out_dim=1, slice_num=32, H=64, W=64, Time_Input=False, unified_pos=False),
    "ONO": dict(space_dim=2, n_layers=10, n_hidden=128, n_head=8, mlp_ratio=2,
                fun_dim=1, out_dim=1, orth=True, psi_dim=32, momentum=0.9,
                attn_type="nystrom", Time_Input=False),
}
