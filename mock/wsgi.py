"""
WSGI config for mock project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mock.settings')

application = get_wsgi_application()

import inspect
from apps.machine_learning_models.registry import MLRegistry
from apps.machine_learning_models.classifier.random_forest import RandomForest

try:
    registry = MLRegistry()
    rf = RandomForest()

    registry.add_algorithm(endpoint_name="classifier",
                            algorithm_object=rf,
                            algorithm_name="random forest",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Carol",
                            algorithm_description="Random forest that computes prediction",
                            algorithm_code=inspect.getsource(RandomForest)
    )

except Exception as e:
    print("Exception while loading the algorithms to the registry",str(e))
