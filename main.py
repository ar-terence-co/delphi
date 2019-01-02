import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import data.stock_data_module as sdm

from modules.Ares.data import AresData as DelphiData
from modules.Ares.ai import AresAI as DelphiAI

#Date Parameters
current_date = None
check = None

#Global data parameters (for viewing)
X_train = None
Y_train = None

X_test = None
Y_test = None

#Global Delphi Managers
stock_manager = sdm.StockDataManager('data')
data_manager = DelphiData(stock_manager)
ai = DelphiAI()

#Hyperparameters
indicators = ['Close','Open','High','Low','EMA-9','EMA-18','EMA-50']
risk = 0.08
reward = 3.0
snapshot_duration = 128

#Check performance of multiple models
train_progress = {}
train_performance = {}
validations_performance = {}
test_performance = {}
predictions = {}

def run():  
    setup()
    run_training()
    run_test()
  
def setup():
    load_new_data()
    load_training_data()
    load_testing_data()    

            
def run_training(model_type, model_name, epochs):
    global X_train, Y_train, X_test, Y_test
    load_delphi(model_type, model_name)
    train_delphi(
            X_train,
            Y_train,
            split=0.8,
            epochs=epochs
        )
    evaluate_delphi(X_test, Y_test)
          
def run_test(model_type, model_name):
    global X_test, Y_test
    load_delphi(model_type, model_name)
    evaluate_delphi(X_test, Y_test)
    predict_with_delphi(X_test, Y_test)

def load_new_data():
    global current_date
    global check
    current_date = datetime.now().strftime('%Y-%m-%d')
    print("Computing today's data: " + current_date)
    
    year = datetime.now().year
    compiler = sdm.StockDataCompiler('data')
    compiler.update_year_range(year - 1,year)
    compiler.save_updates()
    compiler = None
    
    
def load_training_data(
    stock_codes=stock_manager.getBluechips(), 
    start_date=datetime.strptime('2006-12-31','%Y-%m-%d'),
    end_date=datetime.strptime('2016-12-31','%Y-%m-%d'),
): 
    global X_train, Y_train
    global data_manager
    global indicators, risk, reward, snapshot_duration

    print('Loading training data...')
    X_train, Y_train = data_manager.create_dataset(
        stock_codes=stock_codes,
        indicators=indicators,
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
    global indicators, risk, reward, snapshot_duration
 
    print('Loading testing data...')
    X_test, Y_test = data_manager.create_dataset(
        stock_codes=stock_codes,
        indicators=indicators,
        risk=risk,
        reward=reward,
        snapshot_duration=snapshot_duration,
        start_date=start_date,
        end_date=end_date
    )
    
    
def load_delphi(model_type, model_name, filters=[64,64,128,256,512,1024]):
    global ai
    global indicators, snapshot_duration
    
    print('Loading Delphi with "'+model_name+'" configuration...')
    ai.setup(
        model_type=model_type,
        model_name=model_name,
        image_shape=(snapshot_duration, len(indicators)),
        filters=filters
    )
    
    
def train_delphi(
    images,
    labels,
    split=0.7,
    epochs=100
): 
    global ai
    global train_performance, validations_performance, train_progress
    
    train_images, train_labels, eval_images, eval_labels = ai.split_dataset(
        images, labels, split=split
    )        

    print('Training Delphi "'+ai.current_model+'"...')
    train_progress[ai.current_model] = ai.train(train_images, train_labels, epochs=epochs)
    train_performance[ai.current_model] = ai.evaluate(train_images, train_labels)
    validations_performance[ai.current_model] = ai.evaluate(eval_images, eval_labels)
    print('TRAIN DONE')
    return train_performance[ai.current_model].copy(), validations_performance[ai.current_model].copy()
    
    
def evaluate_delphi(images, labels): 
    global ai
    global test_performance   

    print('Evaluating Delphi "'+ai.current_model+'"with test data...')
    test_performance[ai.current_model] = ai.evaluate(images, labels)
    print('EVAL DONE')
    return test_performance[ai.current_model].copy()
    
    
def predict_with_delphi(images,expected_labels = None): 
    global ai
    global predictions

    print('Predicting with Delphi "'+ai.current_model+'"...')
    prediction = ai.predict(images)
    preds = []
    for result in prediction:
        preds.append([result['classes_t0.3'][0],
                      result['classes_t0.5'][0], 
                      result['classes_t0.7'][0], 
                      result['probabilities'][0]])
    preds = np.array(preds)
    if expected_labels is not None:
        preds = np.append(expected_labels,preds,axis=1)
    predictions[ai.current_model] = preds
    
    print('PREDICT DONE')
    return preds.copy()
    

def plot_images(images, labels):
    colors = ['r','b','g','y']
    s = images.shape
    
    t = np.linspace(0, s[1], num=s[1])
    
    plt.figure(figsize=(2*s[2],2*s[0]))
    for i in range(s[0]):
        image = images[i]
        for j in range(0, s[2]):
            plt.subplot(s[0],s[2],i*s[2] + j + 1)
            plt.plot(t, image[:,j], colors[j % len(colors)])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if j == 0:
                plt.ylabel('PROFIT' if labels[i][0] == 1 else 'LOSS')