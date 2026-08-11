from collections import defaultdict, deque


class StateSmoother:
    def __init__(
        self,
        window_size=5,
        required_votes=3,
        transition_frames=3,
    ):
        self.window_size = window_size
        self.required_votes = required_votes
        self.transition_frames = transition_frames

        # Recent raw activity + confidence history
        # stored separately for each track ID.
        self.history = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        # Current stable activity for each person.
        self.current_state = {}

        # Possible new activity waiting to become stable.
        self.candidate_state = {}

        # Number of consecutive updates for which
        # the candidate has remained the winner.
        self.candidate_count = defaultdict(int)

    def update(
        self,
        track_id,
        activity,
        confidence,
    ):
        history = self.history[track_id]

        history.append(
            (activity, confidence)
        )

        # ------------------------------------------
        # Confidence-weighted voting
        # ------------------------------------------

        label_counts = {}
        confidence_scores = {}

        for label, score in history:

            label_counts[label] = (
                label_counts.get(label, 0) + 1
            )

            confidence_scores[label] = (
                confidence_scores.get(label, 0.0)
                + float(score)
            )

        candidate = max(
            confidence_scores,
            key=confidence_scores.get,
        )

        votes = label_counts[candidate]

        # Candidate does not yet have enough
        # observations in the temporal window.
        if votes < self.required_votes:

            return self.current_state.get(
                track_id,
                activity,
            )

        # ------------------------------------------
        # Initialise first stable state
        # ------------------------------------------

        if track_id not in self.current_state:

            self.current_state[track_id] = candidate

            self.candidate_state.pop(
                track_id,
                None,
            )

            self.candidate_count[track_id] = 0

            return candidate

        current = self.current_state[track_id]

        # ------------------------------------------
        # Candidate agrees with stable state
        # ------------------------------------------

        if candidate == current:

            self.candidate_state.pop(
                track_id,
                None,
            )

            self.candidate_count[track_id] = 0

            return current

        # ------------------------------------------
        # Possible state transition
        # ------------------------------------------

        previous_candidate = (
            self.candidate_state.get(track_id)
        )

        if candidate == previous_candidate:

            self.candidate_count[track_id] += 1

        else:

            self.candidate_state[track_id] = (
                candidate
            )

            self.candidate_count[track_id] = 1

        # ------------------------------------------
        # Confirm state transition
        # ------------------------------------------

        if (
            self.candidate_count[track_id]
            >= self.transition_frames
        ):

            self.current_state[track_id] = candidate

            self.candidate_state.pop(
                track_id,
                None,
            )

            self.candidate_count[track_id] = 0

        return self.current_state[track_id]

    def remove(self, track_id):

        if track_id in self.history:
            del self.history[track_id]

        if track_id in self.current_state:
            del self.current_state[track_id]

        if track_id in self.candidate_state:
            del self.candidate_state[track_id]

        if track_id in self.candidate_count:
            del self.candidate_count[track_id]
