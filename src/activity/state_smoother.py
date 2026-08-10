from collections import Counter, defaultdict, deque


class StateSmoother:
    def __init__(
        self,
        window_size=5,
        required_votes=3,
    ):
        self.window_size = window_size
        self.required_votes = required_votes

        self.history = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        self.current_state = {}

    def update(
        self,
        track_id,
        activity,
    ):
        history = self.history[track_id]

        history.append(activity)

        counts = Counter(history)

        candidate, votes = counts.most_common(1)[0]

        if votes >= self.required_votes:
            self.current_state[track_id] = candidate

        return self.current_state.get(
            track_id,
            activity,
        )
