from django.test import TestCase

from apps.machine_learning_models.classifier.random_forest import RandomForest

class MLTests(TestCase):
    def test_rf_algorithm(self):
        data = {
            "Short Desc": "Steel Conduit Hot-Dippedâ Galvanizedâ Steel (Inner Core); PVC (Liquidtightâ Jacket), 1/2 in.",
            "Long Desc": "1/2 in. PVC-coated galvanized steel type ATLA grey liquid-tight conduit. Conduit is 1000 ft."
            # "Short Desc": "Metallic Liquidtight Conduit, Flexible, LA, Galvanized Steel (Inner Core), PVC (Jacket)",
            # "Long Desc": "_x005F_x000D_Type LA_x005F_x000D_",
        }
        algorithm = RandomForest()
        response = algorithm.compute_prediction(data)
        self.assertEqual("OK", response["status"])
        self.assertTrue("label" in response)