#Cleans up stock files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import os

#date sorting key
def getDateFromData(stock_data):
    return stock_data[0]

class StockDataCompiler():
    
    def __init__(self,file_dir):
        self.file_dir = file_dir
        self.inclusive_dates = []
        self.new_dates = []
        self.stock_codes = []
        self.new_stock_codes = []
        self.to_be_added = {}
        
        #Load inclusive dates
        if os.path.isfile(self.file_dir + '/compiled_data/0_inclusive_dates.csv'):
            try:
                inclusive_dates_data = pd.read_csv(self.file_dir + '/compiled_data/0_inclusive_dates.csv')
                self.inclusive_dates = inclusive_dates_data.values[0]
                self.inclusive_dates = self.inclusive_dates[1:len(self.inclusive_dates)].tolist()
            except:
                print('Unreadable 0_inclusive_dates.csv')
                
        #Load stock_codes
        if os.path.isfile(self.file_dir + '/compiled_data/0_stock_codes.csv'):
            try:
                stock_codes_data = pd.read_csv(self.file_dir + '/compiled_data/0_stock_codes.csv')
                self.stock_codes = stock_codes_data.values[0]
                self.stock_codes = self.stock_codes[1:len(self.stock_codes)].tolist()
            except:
                print('Unreadable 0_stock_codes.csv')
                
        

    # STOCKCODE DATE OPEN HIGH LOW CLOSE VOLUME SOME VALUE
    def update_year(self,year):
        update_count = 0
        new_stocks = 0
        earliest_year = 2006
        if year >= earliest_year: #no files before this
            try:
                file_list = os.listdir(self.file_dir + '/raw_data/' + str(year))
            except:
                print('No raw data for ' + str(year))
                return (0,0)
                
            for date_file in file_list: #check all the files
                if date_file.endswith('.csv'):
                    try:
                        dataset = pd.read_csv(self.file_dir + '/raw_data/' + str(year) + '/' + date_file, header = None)
                    except:
                        print(date_file)
                        continue
                    day_date_string = dataset.values[0,1]
                    day_date = datetime.strptime(day_date_string,'%m/%d/%Y')
                    new_day_date_string = day_date.strftime('%Y-%m-%d')
                    
                    
                    if not (new_day_date_string in self.inclusive_dates): #check if the date has already been uploaded
                        update_count = update_count + 1
                        day_data = dataset.values[:,0:8] #if not save the new data
                        day_data[:,1] = [new_day_date_string]*len(day_data[:,1])
                        for stock_day_data in day_data:
                            stock_code = stock_day_data[0]
                            has_dict = stock_code in self.to_be_added
                            stock_data = stock_day_data[1:8]
                            if not has_dict:
                                self.to_be_added[stock_code] = []
                                new_stocks = new_stocks + 1
                                self.new_stock_codes.append(stock_code)
                                
                            self.to_be_added[stock_code].append(stock_data)
                        
                        self.new_dates.append(new_day_date_string)
                        
        print("Year " + str(year) + ": Updated "+str(update_count)+" dates and added " + str(new_stocks)+ " new stocks")
                        
        return (update_count,new_stocks)
                    
    #Batch update
    def update_year_range(self,start_year,end_year):
        print('Updating...')
        update_count = 0
        new_stocks = 0
        for year in range(start_year,end_year+1):
            uc,ns = self.update_year(year)
            update_count = update_count + uc
            new_stocks = new_stocks + ns
            
        print("Update completed: Updated a total of "+str(update_count)+" dates and added " + str(new_stocks) + " new stocks, between " + str(start_year) + " to " + str(end_year))
    
    def save_updates(self):
        print('Saving...')
        errorless = True
        for stock_code in self.to_be_added:
            stock_data = None
            new_stock_data = pd.DataFrame(self.to_be_added[stock_code], columns=['Date','Open','High','Low','Close','Volume','OtherData'])
            new_stock_data['Date'] = pd.to_datetime(new_stock_data['Date'])
            new_stock_data.set_index('Date', inplace=True)
            if stock_code in self.stock_codes:
                stock_data = pd.read_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv')
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                stock_data = pd.concat([stock_data, new_stock_data], sort=False)
            else:
                stock_data = new_stock_data
            stock_data.sort_index(inplace=True)
            
            try:
                stock_data.to_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv',sep = ',')
            except:
                print('Failed to save new '+stock_code + ' data')
                errorless = False
                continue

            print('Saved data to '+stock_code + '.csv')
        if errorless:
            inclusive_dates = self.inclusive_dates + self.new_dates
            sorted(inclusive_dates,key = getDateFromData)
            dates_df = pd.DataFrame([inclusive_dates])

            stock_codes = self.stock_codes + self.new_stock_codes
            codes_df = pd.DataFrame([stock_codes])
            
            try:
                dates_df.to_csv(self.file_dir + '/compiled_data/0_inclusive_dates.csv',sep = ',')
            except:
                print('Failed to save inclusive dates')
                errorless = False
                
            try:
                codes_df.to_csv(self.file_dir + '/compiled_data/0_stock_codes.csv',sep = ',')
            except:
                print('Failed to save stock codes')
                errorless = False
                
        if not errorless:
            print('Please save again')
        else:
            self.inclusive_dates = inclusive_dates
            self.stock_codes = stock_codes
            self.new_dates = []
            self.new_stock_codes = []
            self.to_be_added = {}
            print('Saved successfully')


        
    def __del__(self):
        self.inclusive_dates = None
        self.new_dates = None
        self.stock_codes = None
        self.new_stock_codes = None
        self.to_be_added = None
        print('Closing compiler')
    

class StockDataIndicatorCompiler():
    def __init__(self, file_dir):
        self.file_dir = file_dir
        
        #Load inclusive dates
        if os.path.isfile(self.file_dir + '/compiled_data/0_inclusive_dates.csv'):
            try:
                inclusive_dates_data = pd.read_csv(self.file_dir + '/compiled_data/0_inclusive_dates.csv')
                self.inclusive_dates = inclusive_dates_data.values[0]
                self.inclusive_dates = self.inclusive_dates[1:len(self.inclusive_dates)].tolist()
            except:
                print('Unreadable 0_inclusive_dates.csv')
                
        #Load stock_codes
        if os.path.isfile(self.file_dir + '/compiled_data/0_stock_codes.csv'):
            try:
                stock_codes_data = pd.read_csv(self.file_dir + '/compiled_data/0_stock_codes.csv')
                self.stock_codes = stock_codes_data.values[0]
                self.stock_codes = self.stock_codes[1:len(self.stock_codes)].tolist()
            except:
                print('Unreadable 0_stock_codes.csv')
                
    def update_indicator(self, stock_code, indicator):
        indicator_params = indicator.split('-')
        name = indicator_params[0]
        args = indicator_params[1:]
        
        update_fn = getattr(self, '_update_' + name.lower())
        update_fn(stock_code, *args)
        
                    
    def _update_ema(self, stock_code, period='18'):
        header = 'EMA-' + period
        period = int(period)
        # print('Updating ' + header + ' of ' + stock_code + '...')
        stock_data = pd.read_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv')
        if header not in stock_data or stock_data[header].isnull().values.all():
            stock_data[header] = np.nan
            stock_data.at[period-1, header] = np.mean(stock_data.loc[:(period-1), 'Close'].values)
        multiplier = 2. / (period + 1)
        for i in range(period, len(stock_data.index)):
            if not np.isnan(stock_data.at[i, header]):
                continue
            previous_ema = stock_data.at[i-1, header]
            stock_data.at[i, header] = (stock_data.at[i, 'Close'] - previous_ema)*multiplier + previous_ema 
        try:
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)
            stock_data.to_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv', sep = ',')
        except:
            print('Failed to save ' + stock_code + ' ' + header + ' data')
        finally:
            pass # print('Saved ' + header + ' data to '+stock_code + '.csv')
                
    
#Accessing stocks and indicators
class StockDataManager():
    
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.inclusive_dates = []
        self.stock_codes = []
        self.indicator_compiler = StockDataIndicatorCompiler(file_dir)
        
        #Load inclusive dates
        if os.path.isfile(self.file_dir + '/compiled_data/0_inclusive_dates.csv'):
            try:
                inclusive_dates_data = pd.read_csv(self.file_dir + '/compiled_data/0_inclusive_dates.csv')
                self.inclusive_dates = inclusive_dates_data.values[0]
                self.inclusive_dates = self.inclusive_dates[1:len(self.inclusive_dates)].tolist()
            except:
                print('Unreadable 0_inclusive_dates.csv')
                
        #Load stock_codes
        if os.path.isfile(self.file_dir + '/compiled_data/0_stock_codes.csv'):
            try:
                stock_codes_data = pd.read_csv(self.file_dir + '/compiled_data/0_stock_codes.csv')
                self.stock_codes = stock_codes_data.values[0]
                self.stock_codes = self.stock_codes[1:len(self.stock_codes)].tolist()
            except:
                print('Unreadable 0_stock_codes.csv')
    
    def getStockData(self, stock_code, indicators):
        no_compute = ['Close','Open','High','Low','Volume','Date']
        for indicator in indicators:
            if indicator in no_compute:
                continue
            
            try:
                self.indicator_compiler.update_indicator(stock_code, indicator)
            except:
                raise ValueError(indicator + ' not yet supported')
            
        stock_data = pd.read_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv')
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock = StockData(stock_code, stock_data)
        return stock
    
    def getBluechips(self):
        bluechips = pd.read_csv(self.file_dir + '/bluechips.csv', header=None).values[0].tolist()
        return bluechips
    
class StockData():
    
    INDEX_BEFORE_START = -1
    INDEX_AFTER_END = -2
    
    def __init__(self, stock_code, data):
        self.stock_code = stock_code
        self.data = data
        self.indexToDate = data['Date']
        
    def atIndex(self, index, columns=['Close','Open','High','Low']):
        columns.append('Date')
        if (index >= 0) and (index < len(self.indexToDate)):
            return self.data.loc[index, columns]
        else:
            return None
        
        
    def atIndeces(self, start=None, end=None, index_range=None, columns=['Close','Open','High','Low']):
        columns.append('Date')
        if index_range is not None:
            return self.data.loc[index_range, columns]
        
        if start is not None and end is not None:
            if start == StockData.INDEX_AFTER_END or end == StockData.INDEX_BEFORE_START:
                return None
            if start == StockData.INDEX_BEFORE_START:
                start = 0
            if end == StockData.INDEX_AFTER_END:
                end = len(self.indexToDate)
            if start >= end:
                return None
            index_range = range(start, end)
            return self.data.loc[index_range, columns]
        
        return self.data
    
    
    def atDates(self, start_date, end_date, columns=['Date','Close','Open','High','Low']):
        return self.atIndeces(
            start=self.getIndexFromDate(start_date, for_end_index=False),
            end=self.getIndexFromDate(end_date, for_end_index=True),
            columns=columns
        )
    
        
    def getIndexFromDate(self, this_date, for_end_index=False):
        dates = self.indexToDate.loc[self.indexToDate <= this_date]
        index = len(dates) - 1
        
        if len(self.indexToDate) == 0:
            return StockData.INDEX_AFTER_END
        elif len(dates) == 0:
            return 0 if for_end_index else StockData.INDEX_BEFORE_START
        elif len(dates) == len(self.indexToDate):
            return StockData.INDEX_AFTER_END if for_end_index else len(dates) - 1
        else:
            return index + (1 if for_end_index else 0)
        
        
    def getDateFromIndex(self, index):
        if index < len(self.indexToDate):
            return self.indexToDate[index]
        else:
            return None
