# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np

class AkaiData():
    def __init__(self, manager):
        self.manager = manager
        #date_index = 0
        #close_index = 1
        
    def create_dataset(
            self,
            stock_codes,
            additional_indicators = ['ema-18','ema-50','macd-9-18-50'],
            risk = 0.08,
            reward = 3.0,
            snapshot_duration = 100,
            start_date = datetime.strptime('2005-12-31','%Y-%m-%d'),
            end_date = datetime.now()
    ):
        X_data = []
        Y_data = []
        indicators = ['close','open','high','low'] + additional_indicators
        if start_date >= end_date: 
            return None
        for code in stock_codes:
            snapshots = []
            stock = self.manager.getStockData(code)
            stock.requestData(indicators)
            stock_data = stock.atIndeces(range(
                stock.getIndexFromDate(start_date.strftime('%Y-%m-%d'),roundToPrevious=False) - snapshot_duration,
                stock.getIndexFromDate(end_date.strftime('%Y-%m-%d'),roundToPrevious=True) + 1
            ))
            scores = self._compute_scores(stock_data, risk, reward)[snapshot_duration:]
            for i in range(snapshot_duration - 1,len(stock_data)):
                snapshot_data = stock_data[i - snapshot_duration + 1:i + 1,1:]
                snapshot_data = self._normalize_snapshot(snapshot_data, indicators)
                snapshots.append(snapshot_data)
            for i in range(0,len(scores)):
                if scores[i] != -1:
                    X_data.append(snapshots[i])
                    Y_data.append(scores[i])
        shuffled_indeces = np.arange(0,len(Y_data))
        np.random.shuffle(shuffled_indeces)
        return (np.array(X_data)[shuffled_indeces], np.array(Y_data)[shuffled_indeces])
    
    def _compute_scores(self, stock_data, risk = 0.08, reward = 3.0):
        scores = []
        gain = risk * reward
        closes = stock_data[:,1]
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
        return scores
    
    def _normalize_snapshot(self, snapshot_data, indicators):
        min_value = np.min(snapshot_data)
        max_value = np.max(snapshot_data)
        height = max_value - min_value
        snapshot_data = (snapshot_data - min_value) / height     
        return snapshot_data
            
                
            