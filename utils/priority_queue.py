from queue import PriorityQueue

class DualPriorityQueue(PriorityQueue):
    def __init__(self, maxSize = 10, maxPQ=False):
        PriorityQueue.__init__(self, maxsize=maxSize)
        self.reverse = -1 if maxPQ else 1

    def put(self, data):
        PriorityQueue.put(self, (self.reverse * data[0], data[1]))

    def get(self, *args, **kwargs):
        priority, data = PriorityQueue.get(self, *args, **kwargs)
        return self.reverse * priority, data

