import argparse
import numpy as np
from train import run_training
from predict import run_prediction
from utils import get_latest_data
from data_preprocessing import preprocess_input_data

def main():
    parser = argparse.ArgumentParser(description="Battery Voltage Forecasting")
    parser.add_argument('mode', choices=['train', 'predict'], help="Mode: 'train' or 'predict'")
    parser.add_argument('--data_path', default='data/data.csv', help="Path to the CSV data file")
    parser.add_argument('--model_path', default='model.pth', help="Path to save/load the model")
    parser.add_argument('--n_steps', type=int, default=10, help="Number of steps to predict (default: 10)")

    args = parser.parse_args()

    if args.mode == 'train':
        run_training(args.data_path, args.model_path)
    elif args.mode == 'predict':
        latest_data = get_latest_data(args.data_path)
        print("Latest data shape:", latest_data.shape)

        predictions = run_prediction(args.model_path, latest_data, args.n_steps)
        print("Voltage predictions for the next {} minutes:".format(args.n_steps))
        for i, pred in enumerate(predictions, 1):
            print(f"Minute {i}: {pred:.2f} V")

if __name__ == "__main__":
    main()
