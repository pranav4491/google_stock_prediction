# Google Stock Price Prediction

This project demonstrates the prediction of Google stock prices using a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN) well-suited for time series data. The model is trained on historical stock price data to predict future prices.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Visualization](#data-visualization)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Making Predictions](#making-predictions)
- [Usage](#usage)
- [License](#license)

## Dataset
Download the data from https://finance.yahoo.com
### Training Data
- The training dataset (`GOOG_Train.csv`) contains historical stock prices of Google, including features like Open, High, Low, Close, and Volume.
- The data is parsed by date to ensure proper time series analysis.

### Test Data
- The test dataset (`GOOG_Test.csv`) is used to evaluate the performance of the trained model by comparing the predicted stock prices with actual values.

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `plotly`
  - `scikit-learn`
  - `tensorflow` (Keras)

### Instructions
1. Clone this repository.
2. Ensure that the necessary libraries are installed.
3. Run the code provided in the `google_stock_price_prediction.py` file.

## Data Visualization

Before model training, we visualize the stock price data:

1. **Line Plot**: The High and Low prices are plotted over time to visualize the trends.
2. **Candlestick Chart**: A candlestick chart is used to represent the Open, High, Low, and Close prices over time, providing a more detailed view of price movements.

## Data Preprocessing

1. **Feature Selection**: The model uses the `Open` price as the feature for prediction.
2. **Normalization**: The data is normalized using `MinMaxScaler` to ensure that all values fall within the range of 0 to 1, which is beneficial for LSTM models.
3. **Reshaping**: The training data is reshaped into the required format for LSTM input, i.e., `[samples, timesteps, features]`.

## Model Architecture

The LSTM model is constructed using Keras with the following layers:

1. **LSTM Layer**: The core of the model with 4 units and sigmoid activation.
2. **Dense Layer**: A dense layer with 1 unit to output the predicted stock price.

## Training the Model

- **Loss Function**: The model is compiled using `mean_squared_error` as the loss function.
- **Optimizer**: The Adam optimizer is used to minimize the loss.
- **Training**: The model is trained on the training data for 300 epochs with a batch size of 32.

The loss during training is plotted to visualize the learning process.

## Model Evaluation

- **Test Data Visualization**: The test data is visualized using a candlestick chart to understand the stock price movements during the evaluation period.
- **Prediction**: The model predicts the stock prices for the test set, and these predictions are compared with the actual prices using a line plot.

## Making Predictions

The model predicts Google stock prices for the test set. The predicted prices are then plotted alongside the real prices to visualize the accuracy of the model.

## Usage

To run the prediction:

```python
# Load the test set and preprocess it
test_set = pd.read_csv('/path/to/GOOG_Test.csv', parse_dates=['Date'])
real_stock_price = test_set.iloc[:, 1:2].values

# Normalize and reshape the test data
inputs = sc.transform(real_stock_price)
inputs = np.reshape(inputs, (20, 1, 1))

# Predict the stock prices
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plot the results
plt.figure(figsize=(15,10))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
