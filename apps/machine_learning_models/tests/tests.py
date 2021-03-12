from django.test import TestCase
import inspect

from apps.machine_learning_models.classifier.random_forest import RandomForest
from apps.machine_learning_models.classifier.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        data = {
            "Short Desc": "Steel Conduit Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC (Liquidtightâ Jacket), 1/2 in.",
            "Long Desc": "1/2 in. PVC-coated galvanized steel type ATLA grey liquid-tight conduit. Conduit is 1000 ft."
        }
        algorithm = RandomForest()
        response = algorithm.compute_prediction(data)
        self.assertEqual("OK", response["status"])
        self.assertTrue("label" in response)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "classifier"
        algorithm_object = RandomForest()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Carol"
        algorithm_description = "Random forest that computes prediction"
        algorithm_code = inspect.getsource(RandomForest)

        registry.add_algorithm(endpoint_name, algorithm_object,
                    algorithm_name, algorithm_status, algorithm_version, algorithm_owner, algorithm_description, algorithm_code)

        self.assertEqual(len(registry.endpoints), 1)