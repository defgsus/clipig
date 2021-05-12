from typing import Union, List

import torch


def value_str(
        value: Union[int, float, List[Union[int, float]], torch.Tensor],
        digits: int = 3,
) -> str:
    if isinstance(value, int):
        return str(value)

    elif isinstance(value, float):
        return str(round(value, digits))

    elif isinstance(value, (tuple, list, torch.Tensor)):
        if isinstance(value, torch.Tensor) and not value.shape:
            return str(round(float(value), digits))

        return "[%s]" % (
            ", ".join(value_str(v, digits) for v in value)
        )

    else:
        return str(value)
