import cPickle as pickle
import cleaning_script as cs
import pandas as pd

class MyModel(object):
    def __init__(self, model=None):
        self.model = model

    def load_model(self, filename):
        """
        Loads a new model into the class. The model must be stored in a pickle
        file.

        Parameters
        ----------
        filename: str
            The filename of a pickle file containing the modelmodule

        Returns
        -------
        None
        """
        with open(filename) as f:
            model = pickle.load(f)
            self.model = model

    def transform(self, clean_data):
        """
        Returns a numpy array of data for the model to predict.

        Prepares the cleaned JSON by converting it to a data type that the
        model can use to predict.

        Parameters
        ----------
        clean_data: JSON str
            JSON object of clean data sent by POST request

        Returns
        -------
        prepped_data: Pandas dataframe
            Cleaned data to be used in prediction
        """
        df = pd.read_json(clean_data)
        return df

    def predict(self, prepped_data):
        """
        Returns a probability the event is fraud and a risk label for
        the prepped_data.

        Parameters
        ----------
        prepped_data: numpy array
            Array object of clean data sent by POST request

        Returns
        -------
        fraud_proba: float
            Cleaned data to be used in prediction
        risk_label: str
            Risk level based on thresholds
        """
        fraud_proba = self.model.predict_proba(prepped_data)[0][1]
        if fraud_proba < 0.3:
            risk_label = 'low'
        elif fraud_proba < 0.6 and fraud_proba >= 0.3:
            risk_label = 'medium'
        else:
            risk_label = 'high'
        return fraud_proba, risk_label

    def transform_predict(self, clean_data):
        """
        Returns a probability the event is fraud and a risk label for
        the clean JSON data input.

        Converts the cleaned JSON to a numpy array and then uses the loaded
        model to do a prediction.

        Parameters
        ----------
        clean_data: JSON str
            JSON object of clean data sent by POST request

        Returns
        -------
        fraud_proba: float
            Cleaned data to be used in prediction
        risk_label: str
            Risk level based on thresholds
        """
        data_for_pred = self.transform(clean_data)
        return self.predict(data_for_pred)
