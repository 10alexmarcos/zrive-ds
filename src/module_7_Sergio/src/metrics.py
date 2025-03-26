from time import time
from prometheus_client import Counter, Histogram



class Metrics:
    def __init__(self):
        self.requests = Counter(
            name="service_request",
            documentation="Number of queries to the predict endpoint",
        )

        self.user_not_found_errors = Counter(
            name="user_not_found_errors",
            documentation="Number of errors due to user not found",
        )

        self.model_erros = Counter(
            name="model_errors",
            documentation="Number of errors due to failed model predictions",
        )

        self.unknown_erros = Counter(
            name="unknown_errors", documentation="Number of unknown errors"
        )

        self.predict_duration = Histogram(
            name="predict_duration",
            documentation="Predict duration",
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2.5,
                5,
                10,
                25,
                50,
                100,
                "+Inf",
            ),
        )

    def increase_request(self):
        self.requests.inc()

    def increase_user_not_found_errors(self):
        self.user_not_found_errors.inc()

    def increase_model_errors(self):
        self.model_errors.inc()

    def increase_unknown_errors(self):
        self.unknown_erros.inc()

    def observe_predict_duraction(self, start_time):
        self._observe_time(self.predict_duration, start_time)

    def _calculate_elapsed_time(self, start_time: float):
        return time() - start_time
    
    def _observe_time(self, histogram: Histogram, start_time: float):
        elapsed_time = self._calculate_elapsed_time(start_time)
        histogram.observe(elapsed_time)
