import clip


class ClipSingleton:

    _models = dict()

    @classmethod
    def get(cls, model: str, device: str) -> tuple:
        key = f"{model}/{device}"

        if key not in cls._models:
            cls._models[key] = clip.load(name=model, device=device)

        return cls._models[key]
