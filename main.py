from datamodel import OrderDepth, UserId, TradingState, Order, Product
from typing import List
import string
from sklearn.linear_model import LinearRegression, train_test_split
import numpy as np 
import pandas as pd

prices_day_minus_2 = pd.read_csv('/mnt/data/prices_round_1_day_-2.csv', delimiter=';')
prices_day_minus_1 = pd.read_csv('/mnt/data/prices_round_1_day_-1.csv', delimiter=';')
prices_day_0 = pd.read_csv('/mnt/data/prices_round_1_day_0.csv', delimiter=';')

trades_day_minus_2 = pd.read_csv('/mnt/data/trades_round_1_day_-2_nn.csv', delimiter=';')
trades_day_minus_1 = pd.read_csv('/mnt/data/trades_round_1_day_-1_nn.csv', delimiter=';')
trades_day_0 = pd.read_csv('/mnt/data/trades_round_1_day_0_nn.csv', delimiter=';')

# Concatenate the price and trade data across days
prices_combined = pd.concat([prices_day_minus_2, prices_day_minus_1, prices_day_0])
trades_combined = pd.concat([trades_day_minus_2, trades_day_minus_1, trades_day_0])
prices_combined['avg_bid_price'] = prices_combined[['bid_price_1', 'bid_price_2', 'bid_price_3']].mean(axis=1)
prices_combined['avg_ask_price'] = prices_combined[['ask_price_1', 'ask_price_2', 'ask_price_3']].mean(axis=1)
prices_combined['price_spread'] = prices_combined['avg_ask_price'] - prices_combined['avg_bid_price']
prices_combined['total_bid_volume'] = prices_combined[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1)
prices_combined['total_ask_volume'] = prices_combined[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1)



class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            p1_prices = prices_combined[prices_combined['product'] == product].copy()
            p1_prices['next_mid_price'] = p1_prices['mid_price'].shift(-1)
            p1_prices.dropna(inplace=True)  # Drop rows with NaN values, especially for 'next_mid_price'

            # Selecting features and target for the regression model
            features = p1_prices[['avg_bid_price', 'avg_ask_price', 'price_spread', 'total_bid_volume', 'total_ask_volume']]
            target = p1_prices['next_mid_price']

            # Splitting data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # Training the Linear Regression Model
            model = LinearRegression()
            model.fit(X_train, y_train)
            latest_features = self.extract_latest_features(state, product)
            predicted_mid_price = self.model.predict([latest_features])[0]
            acceptable_price = predicted_mid_price;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    
    def extract_features(state: TradingState, product: Product):
        features = []
    
    # Example feature: Mid-price from observations
        if product in state.observations.plainValueObservations:
            mid_price = state.observations.plainValueObservations[product]
            features.append(mid_price)
    
    # Example feature: Average bid and ask price from conversion observations
        if product in state.observations.conversionObservations:
            conversion_obs = state.observations.conversionObservations[product]
            avg_bid_ask = (conversion_obs.bidPrice + conversion_obs.askPrice) / 2
            features.append(avg_bid_ask)
    
    # Example feature: Total buy and sell order volumes
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            total_buy_volume = sum(order_depth.buy_orders.values())
            total_sell_volume = sum(order_depth.sell_orders.values())
            features.append(total_buy_volume)
            features.append(total_sell_volume)
    
    # Additional features could include recent trade information, environmental factors, etc.
    
        return features
    
    
