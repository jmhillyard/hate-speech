import pandas as pd
import numpy as np
import cleaning as clean
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import log_loss
#import convert_text as ct
import pickle

class Clean_predict(object):
    def __init__(self):
        self.df = clean.load_and_clean('data.json')
        #self.text_features = ct.html_table(df['name'], df['description'])
        self.numeric_df = None
        self.clean_df = None
        self.clean_json_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.json_model = None

    # def add_nlp_features(self):
    #     '''INPUT: matrix features from NLP, cleaned dataframe
    #     Takes cleaned dataframe and adds in matrix features from nlp.
    #     OUTPUT: dataframe for modeling'''
    #     pass

    def feature_engineering(self):
        '''INPUT: json file
        Utilizes clean and load module.
        After loading, file is loaded into panda dataframe and manipulated for feature
        engineering.
        OUTPUT: dataframe for further engineering, does not remove null values'''
        ##create row in dataframe for null value counts in feature engineering
        self.df['null_count'] = self.df.isnull().sum(axis=1)
        ##create new dataframe with only numeric values for FSM
        self.numeric_df = self.df.drop(['acct_type', 'country', 'currency', 'description', 'email_domain', \
            'listed', 'name', 'org_desc', 'org_name', 'payee_name', 'payout_type', \
            'previous_payouts',  'ticket_types', 'venue_address', 'venue_country', \
            'venue_name', 'venue_state', 'has_header', 'sale_duration', 'venue_latitude', 'venue_longitude', 'org_facebook', 'org_twitter' ], axis=1)


    def clean_dataframe(self):
        '''INPUT: numeric_df
        Takes the new dataframe with string entries dropped and deals with null values for
        clean model input.
        OUTPUT: dataframe ready for modeling'''
        #self.clean_df = self.numeric_df.drop(['has_header', 'sale_duration', 'venue_latitude', 'venue_longitude', 'org_facebook', 'org_twitter'], axis=1)
        self.clean_df = self.numeric_df.dropna(axis=0, subset=['delivery_method', 'event_published'])

    def split_data(self):
        '''INPUT: cleaned dataframe of NaN values for modeling
        OUTPUT: Test-train split for modeling purposes'''
        y = self.clean_df['fraud']
        feature_df = self.clean_df.drop('fraud', axis=1)
        X = feature_df
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    def initialize_models(self):
        '''INPUT: None
        Initializes models for fitting and predicting.
        OUTPUT: list of models for fitting'''
        RFC = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                       max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
                       oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,
                       class_weight=None)
        LR = LogisticRegression()
        SVCls = SVC()
        return

    def fit_models(self):
        '''INPUT: training data split
        Takes training data and initializes models to fit to training data.
        OUTPUT: models for prediction'''
        models = initialize_models()
        for each in models:
            each.fit(X_train, y_train)
        return models

    def final_model(self):
        '''INPUT: GBC model
        Fits and predicts GBC model for module.
        OUTPUT: fitted model as json object'''
        model = GradientBoostingClassifier()
        model.fit(self.X_train, self.y_train)
        filename = 'finalized_model.pkl'
        pickle.dump(model, open(filename, 'wb'))

    def model_predict(self, X_test):
        '''INPUT: test data split
        Takes holdout test data and predicts classifications based on the models.
        OUTPUT: numpy matrix of probability predictions'''
        model = final_model()
        self.predictions = model.predict_proba(X_test)
        return

    def model_score(self, y_test, predictions):
        '''INPUT: model predictions and actual values
        OUTPUT: log loss score of model'''
        return log_loss(y_test, predictions)

class Outputs_json(object):
    def __init__(self):
        self.df = df
        self.clean_json_df = clean_json_df
        self.predictions = predictions
        self.final_model = GradientBoostingClassifier()

    def clean_json(self):
        '''INPUT: json object
        Initializes cleaning of dataframe and outputs for server
        OUTPUT: json object of cleaned dataframe'''
        features = Clean_predict.feature_engineering(df)
        numbers_only = Clean_predict.clean_dataframe(features)
        clean_json_df = Clean_predict.numbers_only.to_json()
        return clean_json_df

    def transform_predict(self, clean_json_df):
        pass

if __name__ == '__main__':
    pass
