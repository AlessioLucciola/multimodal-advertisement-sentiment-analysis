
from .utils import *


class SignalAccumulator:
    def __init__(self):
        print(f"Accumulator created!")

    def __call__(self, bvp_queue, pipe):
        self.pipe = pipe
        self.call_back(bvp_queue)

    def call_back(self, queue):
        while True:
            data = self.pipe.recv()
            if data is None:
                break
            bvp = data[0]
            print(f"Received data with length {len(bvp)}")
            for item in bvp:
                queue.put(item)



