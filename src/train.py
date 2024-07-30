import joblib
from data_loader import load_data
from model import train_model
import os

def main():
    data = load_data('data/sample_data.csv')
    X = data[['feature1', 'feature2']]
    y = data['label']
    model = train_model(X, y)
    
    version = os.getenv('MODEL_VERSION', '1.0.0')
    model_filename = f'model_{version}.pkl'
    joblib.dump(model, model_filename)

if __name__ == '__main__':
    main()
