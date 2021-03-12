from django.conf.urls import url, include

from rest_framework.routers import DefaultRouter

from apps.mock_sets.views import (EndpointViewSet, MLAlgorithmViewSet, MLAlgorithmStatusViewSet, MLRequestViewSet, PredictView)

router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    url(r"^api/version1/", include(router.urls)),
    url(
        r"^api/version1/(?P<endpoint_name>.+)/predict$", PredictView.as_view(), name="predict"
    )
]