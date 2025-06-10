import unittest
import pandas as pd
from src.svm.linear_svm import LinearSVM
from src.svm.kernel_svm import KernelSVM
from src.utils.data_loader import load_heart_disease_data, load_news_classification_data

class TestSVMImplementation(unittest.TestCase):

    def setUp(self):
        # Load datasets for testing
        self.X_train, self.X_test, self.y_train, self.y_test = load_heart_disease_data()
        self.news_X_train, self.news_X_test, self.news_y_train, self.news_y_test = load_news_classification_data()

    def test_linear_svm(self):
        model = LinearSVM()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        # Check if predictions are of the correct shape
        self.assertEqual(predictions.shape, self.y_test.shape)
        
        # Check accuracy
        accuracy = (predictions == self.y_test).mean()
        self.assertGreaterEqual(accuracy, 0.7)  # Expecting at least 70% accuracy

    def test_kernel_svm(self):
        model = KernelSVM(kernel='rbf')
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        # Check if predictions are of the correct shape
        self.assertEqual(predictions.shape, self.y_test.shape)
        
        # Check accuracy
        accuracy = (predictions == self.y_test).mean()
        self.assertGreaterEqual(accuracy, 0.75)  # Expecting at least 75% accuracy

    def test_news_classification(self):
        model = KernelSVM(kernel='linear')
        model.fit(self.news_X_train, self.news_y_train)
        predictions = model.predict(self.news_X_test)
        
        # Check if predictions are of the correct shape
        self.assertEqual(predictions.shape, self.news_y_test.shape)
        
        # Check accuracy
        accuracy = (predictions == self.news_y_test).mean()
        self.assertGreaterEqual(accuracy, 0.8)  # Expecting at least 80% accuracy

if __name__ == '__main__':
    unittest.main()