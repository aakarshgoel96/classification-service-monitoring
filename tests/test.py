import unittest
import requests
import base64
from PIL import Image
import io
import numpy as np
import warnings
from typing import List

class TestInferenceService(unittest.TestCase):
    BASE_URL = "http://localhost:8000"  # Update this based on your service URL
    
    def setUp(self):
        """Set up test case with a sample image"""
        # Create a simple test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        self.test_image.save(img_byte_arr, format='PNG')
        self.test_image_bytes = img_byte_arr.getvalue()
        self.test_image_base64 = base64.b64encode(self.test_image_bytes).decode('utf-8')
        
        # Suppress ResourceWarning for unclosed sockets
        warnings.filterwarnings(action="ignore", category=ResourceWarning)
    
    def create_test_images(self, num_images: int) -> List[str]:
        """Create multiple test images for batch testing"""
        images = []
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        for i in range(num_images):
            color = colors[i % len(colors)]
            img = Image.new('RGB', (224, 224), color=color)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            images.append(img_base64)
        return images

    def test_model_info_endpoint(self):
        """Test /info endpoint"""
        response = requests.get(f"{self.BASE_URL}/info")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('name', data)
        self.assertIn('input_shape', data)
        self.assertIn('labels', data)
        
        # Verify model info structure
        self.assertEqual(data['name'], "ResNet50-v2")
        self.assertIsInstance(data['input_shape'], list)
        self.assertIsInstance(data['labels'], list)

    def test_single_prediction(self):
        """Test /predict endpoint with a single image"""
        files = {'file': ('test_image.png', self.test_image_bytes, 'image/png')}
        response = requests.post(f"{self.BASE_URL}/predict", files=files)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('predictions', data)
        predictions = data['predictions']
        
        # Verify prediction structure
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 5)  # Top 5 predictions
        
        for pred in predictions:
            self.assertIn('label', pred)
            self.assertIn('confidence', pred)
            self.assertIsInstance(pred['confidence'], float)
            self.assertTrue(0 <= pred['confidence'] <= 1)

    def test_batch_prediction(self):
        """Test /predict_batch endpoint"""
        batch_size = 3
        test_images = self.create_test_images(batch_size)
        
        payload = {"images": test_images}
        response = requests.post(f"{self.BASE_URL}/predict_batch", json=payload)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('batch_predictions', data)
        predictions = data['batch_predictions']
        
        # Verify batch predictions structure
        self.assertEqual(len(predictions), batch_size)
        
        for batch_pred in predictions:
            self.assertEqual(len(batch_pred), 5)  # Top 5 predictions
            for pred in batch_pred:
                self.assertIn('label', pred)
                self.assertIn('confidence', pred)
                self.assertIsInstance(pred['confidence'], float)
                self.assertTrue(0 <= pred['confidence'] <= 1)

    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        response = requests.get(f"{self.BASE_URL}/metrics")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('resnet_requests_total', response.text)
        self.assertIn('resnet_response_time_seconds', response.text)

    def test_error_handling(self):
        """Test error handling with invalid input"""
        # Test with invalid image data
        files = {'file': ('test.png', b'invalid_image_data', 'image/png')}
        response = requests.post(f"{self.BASE_URL}/predict", files=files)
        self.assertEqual(response.status_code, 500)
        
        # Test batch prediction with invalid base64
        payload = {"images": ["invalid_base64_string"]}
        response = requests.post(f"{self.BASE_URL}/predict_batch", json=payload)
        self.assertEqual(response.status_code, 500)

    def test_performance_requirements(self):
        """Test performance requirements"""
        # Test response time for single prediction
        start_time = time.time()
        files = {'file': ('test_image.png', self.test_image_bytes, 'image/png')}
        response = requests.post(f"{self.BASE_URL}/predict", files=files)
        response_time = time.time() - start_time
        
        self.assertTrue(response_time < 1.0)  # Response time should be under 1 second

if __name__ == '__main__':
    unittest.main(verbosity=2)