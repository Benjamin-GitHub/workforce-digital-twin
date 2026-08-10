from collections import defaultdict, deque


class TemporalPoseBuffer:
    def __init__(self, maxlen=30):
        self.buffers = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, track_id, features):
        self.buffers[track_id].append(features)

    def get(self, track_id):
        return list(self.buffers[track_id])

    def length(self, track_id):
        return len(self.buffers[track_id])

    def remove(self, track_id):
        if track_id in self.buffers:
            del self.buffers[track_id]

    def active_ids(self):
        return list(self.buffers.keys())
