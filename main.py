import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import data.stock_data_module as sdm

from modules.Ares.data import AresData as DelphiData
from modules.Ares.ai import AresAI as DelphiAI

#Date Parameters
current_date = None

#Global data parameters (for viewing)
X_train = None
Y_train = None
train_images = None
train_labels = None
eval_images = None
eval_labels = None

X_test = None
Y_test = None

#Global Delphi Managers
stock_manager = sdm.StockDataManager('data')
data_manager = DelphiData(stock_manager)
ai = DelphiAI()

#Hyperparameters
additional_indicators = []
risk = 0.08
reward = 3.0
snapshot_duration = 100

#Check performance of multiple models
evaluations = {}
predictions = {}

def run():  
    setup()
    run_training()
    run_test()
  
def setup():
    load_new_data()
    load_training_data()
    load_testing_data()    
    
def run_training():
    for i in range(10):
        epochs = i * 2 + 1
        for j in range(5):
            model_name = 'ares_network_'+str(epochs)+'_'+str(j)
            load_delphi(model_name)
            train_delphi(
                split=0.7,
                epochs=epochs
            )
            evaluate_delphi()
          
def run_test():
    global X_test, Y_test
    for i in range(10):
        epochs = i * 2 + 1
        for j in range(5):
            model_name = 'ares_network_'+str(epochs)+'_'+str(j)
            load_delphi(model_name)
            evaluate_delphi()
            predict_with_delphi(X_test, Y_test)

def load_new_data():
    global current_date
    current_date = datetime.now().strftime('%Y-%m-%d')
    print("Computing today's data: " + current_date)
    
    year = datetime.now().year
    compiler = sdm.StockDataCompiler('data')
    compiler.update_year_range(year - 1,year)
    compiler.save_updates()
    compiler = None
    
    
def load_training_data(
    stock_codes=stock_manager.getBluechips(), 
    start_date=datetime.strptime('2011-12-31','%Y-%m-%d'),
    end_date=datetime.strptime('2016-12-31','%Y-%m-%d'),
): 
    global X_train, Y_train
    global data_manager
    global additional_indicators, risk, reward, snapshot_duration

    print('Loading training data...')
    X_train, Y_train = data_manager.create_dataset(
        stock_codes=stock_codes,
        additional_indicators=additional_indicators,
        risk=risk,
        reward=reward,
        snapshot_duration=snapshot_duration,
        start_date=start_date,
        end_date=end_date
    )
  
    
def load_testing_data(
    stock_codes=stock_manager.getBluechips(),
    start_date=datetime.strptime('2017-01-01','%Y-%m-%d'),
    end_date=datetime.now(),
): 
    global X_test, Y_test
    global data_manager
    global additional_indicators, risk, reward, snapshot_duration
 
    print('Loading testing data...')
    X_test, Y_test = data_manager.create_dataset(
        stock_codes=stock_codes,
        additional_indicators=additional_indicators,
        risk=risk,
        reward=reward,
        snapshot_duration=snapshot_duration,
        start_date=start_date,
        end_date=end_date
    )
    
    
def load_delphi(model_name):
    global ai
    global additional_indicators, snapshot_duration
    
    print('Loading Delphi with "'+model_name+'" configuration...')
    ai.setup(
        model_name=model_name,
        image_shape=(snapshot_duration, len(additional_indicators) + 4)
    )
    
    
def train_delphi(
    split=0.7,
    epochs=100
): 
    global X_train, Y_train
    global train_images, train_labels, eval_images, eval_labels
    global ai
    
    train_images, train_labels, eval_images, eval_labels = ai.split_dataset(
        X_train, Y_train, split=split
    )        

    print('Training Delphi "'+ai.current_model+'"...')
    ai.train(train_images, train_labels, epochs=epochs)
    ai.evaluate(eval_images, eval_labels)
    print('TRAIN DONE')
    
    
def evaluate_delphi(): 
    global X_test, Y_test
    global ai
    global evaluations   

    print('Evaluating Delphi "'+ai.current_model+'"with test data...')
    evaluations[ai.current_model] = ai.evaluate(X_test, Y_test)
    print('EVAL DONE')
    
    
def predict_with_delphi(images,expected_labels = None): 
    global ai
    global predictions

    print('Predicting with Delphi "'+ai.current_model+'"...')
    prediction = ai.predict(images)
    
    preds = []
    for result in prediction:
        preds.append([result['classes'][0],result['probabilities'][0]])
    preds = np.array(preds)
    if expected_labels is not None:
        preds = np.append(expected_labels,preds,axis=1)
    predictions[ai.current_model] = preds
    
    print('PREDICT DONE')
    

def plot_training_images(start):
    plt.figure(figsize=(10,10))
    for i in range(start,start+20):
        plt.subplot(1,20,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow((train_images[i]*255).astype(int), cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
