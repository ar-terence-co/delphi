import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, date, time
import data.stock_data_module as sdm
from Akai.model.data import AkaiData as dataProcessor
from Akai.ai.cnn import 
#from Convolvo.ai import ConvolvoAI
#from Convolvo.ui import AviaInterface
#from Convolvo.simulation import ConvolvoSimulation

#import time as T

current_date = ''
stockManager = sdm.StockDataManager('data')

X_data = None
Y_data = None

def run():
    global X_data, Y_data
    setup()
    load_new_data()  
    data = dataProcessor(stockManager)
    X_data, Y_data = data.create_dataset(
        stock_codes = ['ALI', 'SMPH'],
        additional_indicators = [],
        risk = 0.08,
        reward = 3.0,
        snapshot_duration = 100,
        start_date = datetime.strptime('2012-12-31','%Y-%m-%d'),
        end_date = datetime.strptime('2017-12-31','%Y-%m-%d')
    )
    
def setup():
    global current_date
    
    current_date_obj = datetime.now()
    current_date = current_date_obj.strftime('%Y-%m-%d')
    #start_date_obj = current_date_obj - dt.timedelta(days = 425)
    #start_date = start_date_obj.strftime('%Y-%m-%d')

    print("Computing today's data: " + current_date)
    

def load_new_data():
    year = datetime.now().year
    compiler = sdm.StockDataCompiler('data')
    compiler.update_year_range(year - 1,year)
    compiler.save_updates()
    compiler = None