import math
from typing import Callable, Optional

import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


weight_template = 5 * torch.randn((10, 10))

def train(lr: float, iterations: int):
    weights = torch.nn.Parameter(weight_template.clone())
    opt = SGD([weights], lr=lr)

    for t in range(iterations):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())

        loss.backward()
        opt.step()


if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        print(f"Training: {lr=}")
        train(lr, 10)
        print("="*50)

