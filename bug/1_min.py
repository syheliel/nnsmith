import torch
def fn(x):
    return x.max(0).values

fn = torch.compile(fn)
x = torch.randint(0, 100, (16, 2), dtype=torch.int64)
fn(x)