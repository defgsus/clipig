import time
from queue import PriorityQueue
from threading import Thread, Lock
from typing import Any, Tuple

from ..training import ImageTraining


class ImageTrainingThread:

    def __init__(self, out_queue: PriorityQueue):
        self._parameters = None
        self._thread = None
        self._trainer = None
        self._stop = False
        self._in_queue = PriorityQueue()
        self._out_queue = out_queue

    def create(self):
        if not self._thread:
            self._stop = False
            self._thread = Thread(target=self._thread_loop)
            self._thread.start()

    def destroy(self):
        if self._thread:
            self._stop = True
            self.stop_training()
            self._thread.join()
            self._thread = None

    def start_training(self):
        self._put_in_queue("train")

    def stop_training(self):
        if self._trainer and self._trainer._running:
            self._trainer._stop = True
            # TODO: block until not trainer._running

    def _put_in_queue(self, name: str, data: Any = None, priority: int = 10):
        self._in_queue.put(QueueItem(priority, name, data))

    def _put_out_queue(self, name: str, data: Any = None, priority: int = 10):
        self._out_queue.put(QueueItem(priority, name, data))

    def _get_queue(self) -> Tuple[str, Any]:
        item = self._in_queue.get()
        return item.name, item.data

    def set_parameters(self, parameters: dict):
        self._put_in_queue("set_parameters", parameters)

    def _thread_loop(self):
        while not self._stop:

            action_name, data = self._get_queue()
            print("QUEUE", action_name, data)

            if action_name == "set_parameters":
                self._parameters = data
                if not self._trainer:
                    self._trainer = ImageTraining(
                        self._parameters,
                        snapshot_callback=self._snapshot_callback,
                        log_callback=self._log_callback,
                    )
                else:
                    print("SET_PARAMS AGAIN!")
                    raise NotImplementedError

            elif action_name == "train":
                if not self._trainer or not self._trainer._running:
                    if not self._trainer:
                        self._trainer = ImageTraining(
                            self._parameters,
                            snapshot_callback=self._snapshot_callback,
                            log_callback=self._log_callback,
                        )
                    print("TRAIN")
                    self._trainer.train()
                    print("END TRAIN")

        if self._trainer:
            del self._trainer
            self._trainer = None

    def _snapshot_callback(self, tensor):
        self._put_out_queue("snapshot", tensor.detach().cpu())

    def _log_callback(self, *args):
        self._put_out_queue("log", " ".join(str(s) for s in args))


class QueueItem:
    def __init__(self, priority: int, name: str, data: Any):
        self.priority = priority
        self.name = name
        self.data = data

    def __lt__(self, other):
        assert isinstance(other, self.__class__)
        return self.priority < other.priority
