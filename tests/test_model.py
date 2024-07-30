import unittest
from src.model import train_model
import pandas as pd

class TestModel(unittest.TestCase):

    def test_train_model(self):
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [1, 2, 3, 4],
            'label': [0, 1, 0, 1]
        })
        X = data[['feature1', 'feature2']]
        y = data['label']
        model = train_model(X, y)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
