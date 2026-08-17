from django.test import SimpleTestCase
from django.urls import reverse


class APIRouteTests(SimpleTestCase):
    def test_stock_endpoint_exists(self):
        self.assertEqual(reverse('stock-list'), '/api/stocks/')

    def test_dashboard_endpoint_exists(self):
        self.assertEqual(reverse('dashboard', kwargs={'ticker': 'AAPL'}), '/api/dashboard/AAPL/')

    def test_model_metrics_endpoint_exists(self):
        self.assertEqual(reverse('model-metrics', kwargs={'ticker': 'AAPL'}), '/api/model/metrics/AAPL/')
