# --------------------
# /home/zhangzihan/nnsmith/fuzz_report_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-0/model.pth

import numpy as np
import torch
import pickle

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m5 = torch.nn.Conv2d(2, 33, kernel_size=(30, 1), stride=(1, 1), padding=(14, 14))

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        getitem_1 = _args[1];  _args = None
        interpolate = torch.nn.functional.interpolate(getitem, size = [3], scale_factor = None, mode = 'linear', align_corners = None, recompute_scale_factor = None, antialias = False);  getitem = None
        min_1 = torch.min(getitem_1, interpolate);  getitem_1 = interpolate = None
        sin = torch.sin(min_1)
        m5 = self.m5(min_1);  min_1 = None
        to = m5.to(dtype = torch.float32);  m5 = None
        return (sin, to)

m = M()


# Initialize weight
# None

# Initialize input
inp = [v for _, v in pickle.load(open('/home/zhangzihan/nnsmith/fuzz_report_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-0/oracle.pkl', 'rb'))['input'].items()]

# Compile the model
opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)

# Eager run
m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
m_out = [v.cpu().detach() for v in m_out] # torch2numpy
m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out] # torch2numpy

# Compiled run
opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp])
opt_out = [v.cpu().detach() for v in opt_out] # torch2numpy
opt_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in opt_out] # torch2numpy

# Differential testing
for i, (l, r) in enumerate(zip(m_out, opt_out)):
    np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {i}")

# --------------------
