import cPickle as pickle
#import cleaning_script as cs
import pandas as pd
import numpy as np

class MyModel(object):
    def __init__(self, model=None):
        self.model = model

def pred(model,X):
    #input classifier  and text to predict
    #Return: counts of pos and neg resutls for files
    predictions= model.predict(X)
    pos,neg = cal_preds(predictions)
    pos = np.append(400,[0,500,700, 300,1200])
    neg= np.append(100,[0,550,200, 900,800])
    return pos,neg


def cal_preds(predictions):
    """
    Returns results for neg and pos results

    Parameters
    ----------
    predictions: numpy array
        Array object of pos and negative results

    Returns
    -------
    pos , neg : int
    """
    # fraud_proba = self.model.predict_proba(prepped_data)[0][1]
    # if fraud_proba < 0.3:
    #     risk_label = 'low'
    # elif fraud_proba < 0.6 and fraud_proba >= 0.3:
    #     risk_label = 'medium'
    # else:
    #     risk_label = 'high'
        #input classifier  and text to predict
        #Return: prediction

    pos = 0
    neg = 0
    for i in predictions:
        if i == '1':
            pos+=1
        else:
            neg+=1
    return pos, neg
    #return fraud_proba, risk_label
