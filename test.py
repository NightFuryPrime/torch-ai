import pandas as pd
import numpy as np
from ta import add_all_ta_features
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
def preprocess_data(filename):
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d %H:%M:%S')
    df.set_index("timestamp", inplace=True)
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")

    df.fillna(0, inplace=True)
    return df


def create_dataset(data, close_prices, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(close_prices[i])  # Close price
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def main():
    filename = "candlesticks.csv"
    lookback = 60
    test_size = 0.2
    threshold = 0.01  # Define the minimum price difference to open a trade
    df = preprocess_data(filename)
    data = df.iloc[:, 1:].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data[:, 3].reshape(-1, 1)
    scaled_close_prices = close_scaler.fit_transform(close_prices).reshape(-1)
    scaled_data = scaler.fit_transform(data)
    scaled_close_prices = scaled_data[:, 3]

    data_scaled = scaler.fit_transform(data)
    scaled_close_prices = data_scaled[:, 3]

    X, y = create_dataset(data_scaled, scaled_close_prices, lookback)
    y = close_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    print("Original close prices:", close_prices[lookback:lookback+5].reshape(-1))
    print("Close prices in y array:", y[:5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    input_dim = X_train.shape[2]
    hidden_dim = 60
    num_layers = 4
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Add a variable to track the best loss value during training
    best_loss = float('inf')

    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Save the model whenever the loss value improves
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model.pt')
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Load the saved model in subsequent runs
    saved_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    saved_model.load_state_dict(torch.load('best_model.pt'))
    saved_model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float)

    # Replace 'model' with 'saved_model' for evaluation and prediction
    with torch.no_grad():
        y_test_pred = saved_model(X_test_tensor)

    y_test_pred = y_test_pred.detach().numpy()
    y_test = y_test.reshape(-1, 1)
    y_test_pred = close_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(-1)
    y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    # Calculate the difference between predicted and actual prices
    diff = y_test_pred[:-1] - y_test_actual[1:]

    starting_balance = 100.0
    trade_percentage = 0.1
    balance = starting_balance

    open_trade = False
    trade_type = None
    entry_price = 0
    trade_amount = 0
    train_size = int(len(data) * (1 - test_size))
    index_offset = train_size + lookback

    holding_period = 1
    trade_open_time = None
    num_profitable_trades = 0
    total_trades = 0

    for i, delta in enumerate(diff[:-1]):
        if index_offset + i + 1 >= len(df):
            break

        if open_trade:
            holding_period += 1

        if not open_trade:
            if delta > threshold:
                trade_type = 'long'
                entry_price = df['close'].iloc[index_offset + i]
                trade_amount = balance * trade_percentage
                open_trade = True
                trade_open_time = df.index[index_offset + i]
                print(f"Trade at time {trade_open_time}: Buy at {entry_price}")
            elif delta < -threshold:
                trade_type = 'short'
                entry_price = df['close'].iloc[index_offset + i]
                trade_amount = balance * trade_percentage
                open_trade = True
                trade_open_time = df.index[index_offset + i]
                print(f"Trade at time {trade_open_time}: Sell at {entry_price}")

        if open_trade:
            trade_close_time = df.index[index_offset + i + 1]
            if trade_type == 'long':
                sell_price = df['close'].iloc[index_offset + i + 1]
                profit = sell_price - entry_price
                if profit > threshold:
                    balance += trade_amount * (sell_price - entry_price) / entry_price
                    open_trade = False
                    trade_type = None
                    
                    holding_period = (trade_close_time - trade_open_time).total_seconds()/60
                    print(f"Trade at time {trade_close_time}: Sell at {sell_price}, holding period: {holding_period} minutes")
                    if profit > 0:
                        num_profitable_trades += 1
                    total_trades += 1
                    holding_period = 1

            else:  # trade_type == 'short'
                buy_back_price = df['close'].iloc[index_offset + i + 1]
                profit = entry_price - buy_back_price
                if profit > threshold:
                    balance += trade_amount * (entry_price - buy_back_price) / entry_price
                    open_trade = False
                    trade_type = None
                    
                    holding_period = (trade_close_time - trade_open_time).total_seconds()/60
                    print(f"Trade at time {trade_close_time}: Buy back at {buy_back_price}, holding period: {holding_period} minutes")
                    if profit > 0:
                        num_profitable_trades += 1
                    total_trades += 1
                    holding_period = 1
        
    print(f"Final balance: {balance}")

    win_rate = num_profitable_trades / total_trades * 100
    print(f"Win rate: {win_rate:.2f}%")

if __name__ == "__main__":
    main()
