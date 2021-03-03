from apps.mock_sets.models import Endpoint
from apps.mock_sets.models import MLAlgorithm
from apps.mock_sets.models import MLAlgorithmStatus

class MLRegistry:
    def __init__(self):
        self.endpoints = {}

    def add_algorithm(
            self, endpoint_name, algorithm_object, algorithm_name, algorithm_status, algoritm_version, owner):