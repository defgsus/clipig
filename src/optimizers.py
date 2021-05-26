from typing import Tuple

import torch
import torch.nn


def create_optimizer(model: torch.nn.Module, params: str) -> Tuple[float, torch.optim.Optimizer]:

    if params == "adam":
        
        base_learnrate = 0.01
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_learnrate,  # will be adjusted per epoch
        )
        
    elif params == "sgd":
        base_learnrate = 10.0
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_learnrate,  # will be adjusted per epoch
        )

    elif params == "sparse_adam":
        base_learnrate = 0.01
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=base_learnrate,  # will be adjusted per epoch
        )

    elif params == "adadelta":
        base_learnrate = 20.0
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=base_learnrate,  # will be adjusted per epoch
        )

    elif params == "rmsprob":
        base_learnrate = 0.003
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=base_learnrate,  # will be adjusted per epoch
            centered=True,
            # TODO: high momentum is quite useful for more 'chaotic' images but needs to
            #   be adjustable by config expression
            momentum=0.1,
        )
    else:
        raise ValueError(f"Unknown optimizer '{params}'")

    return base_learnrate, optimizer
