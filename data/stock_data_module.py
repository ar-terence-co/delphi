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
            file_list = os.listdir(self.file_dir + '/raw_data/' + str(year))
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
            stock_data = []
            if stock_code in self.stock_codes:
                stock_data = pd.read_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv').values[:,1:8]
                stock_data = np.concatenate((stock_data,np.asarray(self.to_be_added[stock_code])),axis = 0)
            else:
                stock_data = self.to_be_added[stock_code]
                
            sorted(stock_data,key = getDateFromData)
            df = pd.DataFrame(stock_data,columns = ['Date','Open','High','Low','Close','Volume','OtherData'])
            try:
                df.to_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv',sep = ',')
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
    


#Accessing stocks and indicators
class StockDataManager():
    
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.inclusive_dates = []
        self.stock_codes = []
        
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
    
    def getStockData(self, stock_code):
        stock_data = pd.read_csv(self.file_dir + '/compiled_data/' + stock_code + '.csv').values[:,1:8]
        stock = StockData(stock_code,stock_data)
        return stock
    
class StockData():
    
    def __init__(self, stock_code, data):
        self.stock_code = stock_code
        self.data = data
        self.r_data = self.data
        self.indexToDate = data[:,0].T
        self.r_indexToDate = self.indexToDate
        
    def atIndex(self, index):
        if (index >= 0) and (index < len(self.r_indexToDate)):
            return self.r_data[index]
        else:
            return False
        
    def atIndeces(self, index_range):
        return self.r_data[index_range]
        
    def getIndexFromDate(self, datestring, roundToPrevious):
        if datestring in self.r_indexToDate:
            index = np.where(self.r_indexToDate == datestring)[0][0]
            return index
        else:
            offset = 1
            if roundToPrevious:
                offset = -1
            date_object = datetime.strptime(datestring,'%Y-%m-%d')
            one_day_offset = timedelta(offset)
            if roundToPrevious:
                limit_date_object = datetime.strptime(self.r_indexToDate[0],'%Y-%m-%d')
            else:
                limit_date_object = datetime.strptime(self.r_indexToDate[len(self.r_indexToDate)-1],'%Y-%m-%d')
            
            while (roundToPrevious and date_object > limit_date_object) or ((not roundToPrevious) and date_object < limit_date_object):
                date_object = date_object + one_day_offset
                offseted_datestring = date_object.strftime('%Y-%m-%d')
                if offseted_datestring in self.r_indexToDate:
                    index = np.where(self.r_indexToDate == offseted_datestring)[0][0]
                    return index
                
            return -1
        
    def getDateFromIndex(self, index):
        if index < len(self.r_indexToDate):
            return self.r_indexToDate[index]
        
    #o = open prices
    #h = high prices
    #l = low prices
    #c = close prices
    #volume = volume
    #percent_delta = percent change from previous close price
    #sma-10 = where 10 can be any int, SMA for the given period
    #ema-10 = where 10 can be any int, EMA for the given period
    #macd-9-12-26 = where 9,12,26 can be any combination of int in ascending order, first number is the signal line and 12-26 is the MACD
    
    def requestData(self,request = ['open','high','low','close','volume','percent_delta',
                'ema-18','ema-50','macd-9-18-50']):
        offset = 0
        data_dict = {}
        for requestStr in request:
            requestString = requestStr
            if not isinstance(requestString, str):
                continue
            else:
                requestString = requestString.lower()
            
            if requestString == 'open':
                if not data_dict.has_key('open'):
                    data_dict['open'] = self.data[:,1].T
            elif requestString == 'high':
                if not data_dict.has_key('high'):
                    data_dict['high'] = self.data[:,2].T            
            elif requestString == 'low':
                if not data_dict.has_key('low'):
                    data_dict['low'] = self.data[:,3].T 
            elif requestString == 'close':
                if not data_dict.has_key('close'):
                    data_dict['close'] = self.data[:,4].T 
            elif requestString == 'volume':
                if not data_dict.has_key('volume'):
                    data_dict['volume'] = self.data[:,5].T             
            elif requestString == 'percent_delta':
                if offset < 1:
                    offset = 1
                if not data_dict.has_key('percent_delta'):
                    if not data_dict.has_key('close'):
                        data_dict['close'] = self.data[:,4].T                   
                    data_dict['percent_delta'] = self.getPercentDelta(data_dict['close'])
            elif requestString.startswith('ema-'):
                if not data_dict.has_key(requestString):
                    numStr = requestString.split('ema-')[1]
                    if numStr.isdigit():
                        n = int(numStr)
                        if not (n == 0) and (len(self.data) >= n):
                            if offset < n:
                                offset = n
                            if not data_dict.has_key('close'):
                                data_dict['close'] = self.data[:,4].T                   
                            data_dict[requestString] = self.getEMA(n,data_dict['close'])
                        else:
                            print 'Not valid period'
                    else:
                        print 'Please use format "ema-40"'
            elif requestString.startswith('sma-'):
                if not data_dict.has_key(requestString):
                    numStr = requestString.split('sma-')[1]
                    if numStr.isdigit():
                        n = int(numStr)
                        if not (n == 0) and (len(self.data) >= n):
                            if offset < n:
                                offset = n
                            if not data_dict.has_key('close'):
                                data_dict['close'] = self.data[:,4].T                   
                            data_dict[requestString] = self.getSMA(n,data_dict['close'])
                        else:
                            print 'Not valid period'
                    else:
                        print 'Please use format "sma-40"'
            elif requestString.startswith('macd-'):
                if not data_dict.has_key(requestString):
                    numStrArr = requestString.split('-')
                    if len(numStrArr) == 4:
                        if numStrArr[1].isdigit() and numStrArr[2].isdigit() and numStrArr[3].isdigit():
                            signal = int(numStrArr[1])
                            lesser = int(numStrArr[2])
                            greater = int(numStrArr[3])
                            if (signal < lesser) and (greater > lesser):
                                if not (signal == 0) and (len(self.data) >= greater):
                                    if offset < (greater + signal - 1):
                                        offset = (greater + signal - 1)
                                    if not data_dict.has_key('close'):
                                        data_dict['close'] = self.data[:,4].T
                                    if not data_dict.has_key('ema-'+numStrArr[2]):
                                        data_dict['ema-'+numStrArr[2]] = self.getEMA(lesser,data_dict['close'])
                                    if not data_dict.has_key('ema-'+numStrArr[3]):
                                        data_dict['ema-'+numStrArr[3]] = self.getEMA(greater,data_dict['close'])
                                        
                                    data_dict[requestString] = self.getMACD([signal,lesser,greater],
                                             data_dict['ema-'+numStrArr[2]],
                                             data_dict['ema-'+numStrArr[3]])
                                else:
                                    print 'Not valid periods'
                            else:
                                print 'Periods should be ascending'
                        else:
                            print 'Please use format"macd-9-12-26"' 
                    else:
                            print 'Please use format"macd-9-12-26"' 

        returnArray = [self.data[:,0].T] 
        for requestStr in request:
            requestString = requestStr
            if not isinstance(requestString, str):
                continue
            else:
                requestString = requestString.lower()
            
            if data_dict.has_key(requestString):
                returnArray.append(data_dict[requestString])
        
        returnArray = np.array(returnArray)
        offsetedArray = returnArray[:,offset:len(self.data)]
        self.r_data = offsetedArray.T
        self.r_indexToDate = self.r_data[:,0].T

    def getPercentDelta(self,closing_prices):
        previous_price = closing_prices[0]
        percent_delta_array = [-100]
        for i in range(1,len(closing_prices)):
            percent_delta = -100            
            if not (previous_price == 0):
                percent_delta = (closing_prices[i] - previous_price)/previous_price
            
            percent_delta_array.append(percent_delta)
            previous_price = closing_prices[i]
        return np.array(percent_delta_array)
    
    def getEMA(self,period,closing_prices):
        starting_prices = closing_prices[0:period]
        previous_ema = sum(starting_prices)/period
        multiplier = 2 / (period + 1.)
        ema_array = [previous_ema]*period
        for i in range(period,len(closing_prices)):
            ema = (closing_prices[i] - previous_ema)*multiplier + previous_ema
            ema_array.append(ema)
        return np.array(ema_array)
    
    def getSMA(self,period,closing_prices):
        starting_prices = closing_prices[0:period]
        previous_sma = sum(starting_prices)/period
        sma_array = [previous_sma]*period
        for i in range(period,len(closing_prices)):
            sma = sum(closing_prices[i-period+1:i+1])/period
            sma_array.append(sma)
        return np.array(sma_array)
    
    def getMACD(self,periods,lesserEMA,greaterEMA):
        greater = periods[2]
        signal = periods[0]
        previous_macd = lesserEMA[greater-1] - greaterEMA[greater-1]
        macd_array = [previous_macd]*greater
        for i in range(greater,len(greaterEMA)):
            macd = lesserEMA[i] - greaterEMA[i]
            macd_array.append(macd)
        signalLine = self.getEMA(signal,np.array(macd_array)[greater-1:len(macd_array)])
        
        previous_macd_hist = macd_array[greater+signal-2] - signalLine[signal-1] 
        macd_hist_array = [previous_macd_hist]*(greater+signal-1)
        for i in range(greater+signal-1,len(macd_array)):
            macd_hist = macd_array[i] - signalLine[i-greater+1] 
            macd_hist_array.append(macd_hist)

        return np.array(macd_hist_array)


            

