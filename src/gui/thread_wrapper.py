import time
from queue import Queue
from threading import Thread, Lock
from typing import Any, Tuple, Optional

import PIL.Image
from torchvision.transforms.functional import to_pil_image

from ..training import ImageTraining


class ImageTrainingThread:

    def __init__(self, out_queue: Queue):
        self._parameters = None
        self._thread = None
        self._trainer: Optional[ImageTraining] = None
        self._stop = False
        self._in_queue = Queue()
        self._out_queue = out_queue
        self._last_image_tensor = None
        self._running_counts = dict()

    def create(self):
        """Start the thread"""
        if not self._thread:
            self.log("starting thread")
            self._stop = False
            self._thread = Thread(target=self._thread_loop)
            self._thread.start()

    def destroy(self):
        """Destroy the thread"""
        if self._thread:
            self.log("stopping thread")
            self._stop = True
            self._put_in_queue("-")  # just put something in the queue to end the loop
            self.pause_training()
            self._thread.join()
            self._thread = None

    def is_running(self) -> bool:
        return self._trainer and self._trainer._running

    def start_training(self, parameters: dict):
        """Start (or restart) a training"""
        self._put_in_queue("start", parameters)

    def pause_training(self):
        """Pause the current session"""
        if self._trainer and self._trainer._running:
            self.log("pause training")
            self._trainer._stop = True
            # i know, this is not the best method...
            while self._trainer and self._trainer._running:
                time.sleep(.1)

    def continue_training(self):
        """Continue training if possible"""
        if not self._parameters:
            return
        self._put_in_queue("continue")

    def update_parameters(self, parameters: dict):
        if not self._trainer:
            self.start_training(parameters)
            return

        self.pause_training()
        parameters = parameters.copy()
        parameters["start_epoch"] = self._trainer.epoch
        if self._last_image_tensor is not None:
            parameters["init"] = {
                "image": None,
                "image_tensor": self._last_image_tensor.tolist(),
                "mean": [0, 0, 0],
                "std": [0, 0, 0],
            }

        self._put_in_queue("update_parameters", parameters)
        self._put_in_queue("continue")

    def get_image(self) -> Optional[PIL.Image.Image]:
        if self._last_image_tensor is not None:
            return to_pil_image(self._last_image_tensor)

    def log(self, *args):
        self._put_out_queue("log", " ".join(str(a) for a in args))

    def running_counts(self) -> dict:
        return self._running_counts.copy()

    def _put_in_queue(self, name: str, data: Any = None, priority: int = 10):
        self._in_queue.put(QueueItem(priority, name, data))

    def _put_out_queue(self, name: str, data: Any = None, priority: int = 10):
        self._out_queue.put(QueueItem(priority, name, data))

    def _get_queue(self) -> Tuple[str, Any]:
        item = self._in_queue.get()
        return item.name, item.data

    def _thread_loop(self):
        while not self._stop:
            try:
                action_name, data = self._get_queue()
                # print("QUEUE", action_name, data)

                if action_name == "start":
                    if self._trainer:
                        self.pause_training()
                        self._destroy_trainer()
                    self._parameters = data
                    self._running_counts = dict()
                    self.continue_training()

                elif action_name == "continue":
                    if not self._trainer:
                        self._create_trainer()
                    else:
                        self._update_trainer_running_counts()
                    self.log("start training loop")
                    self._put_out_queue("started")
                    self._trainer.train()
                    self._put_out_queue("stopped")
                    self._snapshot_callback(self._trainer.pixel_model.forward())
                    self.log("training loop ended")

                elif action_name == "update_parameters":
                    if self._trainer:
                        self.pause_training()
                        self._destroy_trainer()
                    self._parameters = data

            except Exception as e:
                self.log(f"{type(e).__name__}: {e}")

        if self._trainer:
            del self._trainer
            self._trainer = None

    def _create_trainer(self):
        self._trainer = ImageTraining(
            self._parameters,
            snapshot_callback=self._snapshot_callback,
            log_callback=self._log_callback,
            progress_callback=self._progress_callback,
        )
        self._update_trainer_running_counts()

    def _update_trainer_running_counts(self):
        for key in ("training_seconds", "forward_seconds", "backward_seconds"):
            if key in self._running_counts:
                setattr(self._trainer, key, self._running_counts[key])

    def _destroy_trainer(self):
        del self._trainer
        self._trainer = None

    def _snapshot_callback(self, tensor):
        cpu_tensor = tensor.detach().cpu()
        self._last_image_tensor = cpu_tensor
        self._put_out_queue("snapshot", cpu_tensor)

    def _log_callback(self, *args):
        self._put_out_queue("log", " ".join(str(s) for s in args))

    def _progress_callback(self, stats: dict):
        for key in ("training_seconds", "forward_seconds", "backward_seconds"):
            self._running_counts[key] = stats[key]
        self._running_counts["training_epochs"] = self._running_counts.get("training_epochs", 0) + 1

        self._put_out_queue("progress", stats)


class QueueItem:
    def __init__(self, priority: int, name: str, data: Any):
        self.priority = priority
        self.name = name
        self.data = data

    def __lt__(self, other):
        assert isinstance(other, self.__class__)
        return self.priority < other.priority
