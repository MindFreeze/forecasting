# Battery Voltage Forecasting

This project implements a transformer-based neural network model to forecast battery voltage based on historical data.

## Setup

1. Install the required packages:

```sh
pip install -r requirements.txt
```

2. Ensure your data is in the `data/data.csv` file.

## Usage

To train the model:

```sh
python main.py train
```

To make predictions:

```sh
python main.py predict
```

You can specify custom paths for data and model files using the `--data_path` and `--model_path` arguments. Use `--n_steps` to specify the number of minutes to predict (default is 10).

For more information, run:

```sh
python main.py --help
```
