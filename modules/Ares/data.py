# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np

class AresData():
    def __init__(self, manager):
        self.manager = manager
        #date_index = 0
        #close_index = 1
        
    def create_dataset(
            self,
            stock_codes,
            indicators = ['Close','Open','High','Low','EMA-18','EMA-50','EMA-100'],
            risk = 0.08,
            reward = 3.0,
            snapshot_duration = 100,
            start_date = datetime.strptime('2005-12-31','%Y-%m-%d'),
            end_date = datetime.now(),
            keep_incomplete = False
    ):
        X_data = []
        Y_data = []
        if 'Close' not in indicators:
            indicators.append('Close')
        if start_date >= end_date: 
            return None
        for code in stock_codes:
            snapshots = []
            stock = self.manager.getStockData(code, indicators)
            stock_data = stock.atDates(start_date, end_date, indicators)
            stock_data = stock_data.dropna()
            stock_data = stock_data.drop('Date', axis='columns')
            scores = self._compute_scores(stock_data, risk, reward)[snapshot_duration:]
            stock_values = stock_data.values
            for i in range(snapshot_duration - 1,len(stock_values)):
                snapshot_data = stock_values[i - snapshot_duration + 1:i + 1,:]
                snapshot_data = self._normalize_snapshot(snapshot_data)
                snapshots.append(snapshot_data)
            for i in range(0,len(scores)):
                if scores[i] != -1:
                    X_data.append(snapshots[i])
                    Y_data.append(scores[i])
        return (np.array(X_data).astype(float), np.array(Y_data).reshape(len(Y_data),1))
    
    def _compute_scores(self, stock_data, risk = 0.08, reward = 3.0):
        scores = []
        gain = risk * reward
        closes = stock_data['Close'].values
        for i in range(0,len(closes)):
            take_profit = closes[i]*(1+gain)
            stop_loss = closes[i]*(1-risk)
            stopped = False
            for j in range(i,len(closes)):
                if closes[j] >= take_profit:
                    scores.append(1)
                    stopped = True
                    break
                elif closes[j] <= stop_loss:
                    scores.append(0)
                    stopped = True
                    break
            if not stopped:
                scores.append(-1)
        return scores #ndarray
    
    def _normalize_snapshot(self, snapshot_data):
        min_value = np.min(snapshot_data)
        max_value = np.max(snapshot_data)
        height = max_value - min_value
        snapshot_data = (snapshot_data - min_value) / height     
        return snapshot_data
            
                
            