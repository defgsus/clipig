import torch
import clip


class ClipSingleton:

    _models = dict()

    @classmethod
    def get(cls, model: str, device: str) -> tuple:
        """
        Return CLIP model and preprocessor
        :param model: str, name
        :param device: str, a torch device or 'auto'
        :return: tuple of (Module, Module)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Cuda device requested but not available")

        key = f"{model}/{device}"

        if key not in cls._models:
            cls._models[key] = clip.load(name=model, device=device)

        return cls._models[key]
