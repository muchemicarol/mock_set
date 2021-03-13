from apps.mock_sets.models import Endpoint
from apps.mock_sets.models import MLAlgorithm
from apps.mock_sets.models import MLAlgorithmStatus

from apps.machine_learning_models.classifier.random_forest import RandomForest

import inspect

class MLRegistry:
    def __init__(self):
        self.endpoints = {}

    def add_algorithm(
            self, endpoint_name, algorithm_object, algorithm_name, algorithm_status, algoritm_version, owner, algorithm_description, algorithm_code):

        # get endpoint
        endpoint, _ = Endpoint.objects.get_or_create(
                        name=endpoint_name, owner=owner)

        # get algorithm
        database_object, algorithm_created = MLAlgorithm.objects.get_or_create(
            name=algorithm_name,
            description=algorithm_description,
            code=algorithm_code,
            version=algoritm_version,
            owner=owner,
            parent_endpoint=endpoint
        )

        if algorithm_created:
            status = MLAlgorithmStatus(
                status=algorithm_status,
                created_by=owner,
                parent_mlalgorithm=database_object,
                active=True
            )

            status.save()

        # add to registry
        self.endpoints[database_object.id] = algorithm_object

registry = MLRegistry()
rf = RandomForest()
registry.add_algorithm(endpoint_name="classifier",
                            algorithm_object=rf,
                            algorithm_name="random forest",
                            algorithm_status="production",
                            algoritm_version="0.0.1",
                            owner="Carol",
                            algorithm_description="Random forest that computes prediction",
                            algorithm_code=inspect.getsource(RandomForest)
    )