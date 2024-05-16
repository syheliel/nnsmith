import numpy as np
import torch

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v10_0 = torch.nn.Parameter(torch.empty([1, 1, 1, 1], dtype=torch.complex64), requires_grad=False)
        self.v8_0 = torch.nn.Parameter(torch.empty([17, 1, 1, 12, 5], dtype=torch.complex64), requires_grad=False)
        self.v5_0 = torch.nn.Parameter(torch.empty([17, 1, 1, 1, 5], dtype=torch.complex64), requires_grad=False)

    def forward(self, getitem, getitem_1):
        add = torch.add(getitem, self.v10_0); 
        cat = torch.cat((self.v5_0, add, getitem_1, self.v8_0), dim = 3)
        expand = cat.expand(17, 1, 1, 15, 5)
        return expand

m = M()

# Initialize input
arg1 = torch.randint(0, 100, (17, 1,1,1,5), dtype=torch.int64) + 1j 
arg2 = torch.randint(0, 100, (17, 1,1,1,5), dtype=torch.int64) + 1j
opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)
eager_out = m(arg1,arg2)
opt_out = opt(arg1,arg2)
