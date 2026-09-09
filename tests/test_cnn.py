import unittest
import numpy as np
from src.cnn_classifier import EyeCNNClassifier, build_eye_cnn

class TestEyeCNN(unittest.TestCase):
    def test_cnn_architecture(self):
        model = build_eye_cnn(input_shape=(64, 64, 3))
        self.assertEqual(model.output_shape, (None, 1))

    def test_cnn_inference(self):
        classifier = EyeCNNClassifier()
        dummy_crop = np.zeros((64, 64, 3), dtype=np.float32)
        state, prob = classifier.predict_eye(dummy_crop)
        self.assertIn(state, ["Open", "Closed"])
        self.assertTrue(0.0 <= prob <= 1.0)

if __name__ == "__main__":
    unittest.main()
