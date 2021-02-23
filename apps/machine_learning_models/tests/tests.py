from django.test import TestCase

from apps.machine_learning_models.classifier.logistic_regression import LogisticRegression

class MLTests(TestCase):
    def test_lr_algorithm(self):
        data = {
            "Short Desc": "Metallic Liquidtight Conduit, Flexible, LA, Galvanized Steel (Inner Core), PVC (Jacket)",
            "Long Desc": "_x000D_Type LA_x000D_",
            "Size": "2-1/2 in.",
            "Length": "100 ft.",
            "Material": "PVC Coated Galvanized Steel"
        }
        algorithm = LogisticRegression()
        response = algorithm.preprocessing(data)
        self.assertEqual("OK", response["status"])