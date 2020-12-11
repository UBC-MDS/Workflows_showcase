"""
Author: Aidan Mattrick

Date: Nov 28, 2020

This file defines a predictive model class library for this project.
"""

import os
import re
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier

from tqdm import tqdm

class CharacterPredictiveModel:
    """A character model predictor function class.

    Parameters
    ----------
    verbose : boolean
        A boolean flag determining if verbose output is enabled.

    Examples
    --------
    >>> cpm = CharacterPredictiveModel(false)
    >>> cpm.fit(X_train, y_train)
    >>> cpm.optimize_model(model)
    >>> cpm.return_model()
    """

    def __init__(self, verbose = False):
        """See help(CharacterPredictiveModel)"""
        self.verbose = verbose

    def fit(self, X, y):
        """
        Reads input dataframe and creates preprocessor
        Parameters:
        -----------
        input_df : pandas DataFrame
            input feature and target data

        Returns:
        -----------
        """
        self.X = X
        self.y = y
        self.preprocessor = self.build_preprocessor()

        if self.verbose: print('Fit input features and targets.')
        return


    def store_cross_val_results(self, model, X_train, y_train,
                                scoring_metric = "accuracy"):
        """
        Returns mean and std of cross validation.
        This function is adapted from Varada's lecture code in DSCI571

        Parameters
        ----------
        model :
            scikit-learn classification model
        X_train : DataFrame
            X Training data, indepedent variables
        y_train : DataFrame
            Training data, dependent variables
        scoring_metric: string
            Metric to use for scoring

        Returns
        ----------
            Dict
        """
        scores = cross_validate(model,
                                X_train, y_train,
                                return_train_score=True,
                                n_jobs=-1,
                                scoring=scoring_metric);
        mean_scores = pd.DataFrame(scores).mean()
        std_scores = pd.DataFrame(scores).std()
        out_col = []

        # Add standard deviation to 3 decimal places to score report
        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

        return pd.Series(data = out_col, index = mean_scores.index)


    def build_preprocessor(self):
        """
        Builds the model preprocessor pipeline

        Parameters
        ----------

        Returns
        ----------
            sklearn Pipeline object
        """       

        numeric_features = ['appearances', 'year', 'name_len', 'appear_per_yr']
        categorical_features = ['id', 'eye', 'hair', 'publisher', 'sex']
        drop_features = ['name', 'gsm', 'first_appearance']
        binary_features = ['is_common', 'has_last_name']

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        binary_transformer = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown='error', drop='if_binary', dtype=int)),
            ]
        )

        # Assemble preprocessor pipeline
        preprocessor = make_column_transformer(
            ("drop", drop_features),
            (numeric_transformer, numeric_features),
            (categorical_transformer, categorical_features),
            (binary_transformer, binary_features),
        )
        return preprocessor


    def model_compare(self, models):
        """
        Processes the data, trains the model, and returns a dataframe showing the results

        Parameters
        ----------
        models : dict
            Models to be trained

        Returns
        ----------
            pandas.Dataframe

        """
        assert(self.X.empty is not False and self.y.empty is not False, 
               "Error, fit model before compare")
        results_df = {}

        for key, value in tqdm(models.items()):
            model_pipe = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    (key, value)
                ]
            )
            results_df[key] = self.store_cross_val_results(model_pipe, 
                                                           self.X, 
                                                           self.y)

        results = pd.DataFrame(results_df)
        results.reset_index(inplace=True)
        results = results.rename(columns = {'index':'Scores'})
        #save_img(results, output_folder_path, filename)
        return results


    def optimize_model(self, model, param_grid=None):
        """
        Processes the data, performs hyperparameter optimization on the model, and returns that model

        Parameters
        ----------
        model : dict
            Model name key and sklearn classifier value

        param_grid : dict
            hyperparameters to optimize model

        fileprefix : string
            Prefix to be added to the filename of the output

        Returns
        ----------
            pandas.Dataframe

        """
        assert(self.X.empty is not False and self.y.empty is not False, 
               "Error, fit model before compare")
        optimized_model_pipe = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (list(model.keys())[0], list(model.values())[0])
            ]
        )

        random_search = RandomizedSearchCV(optimized_model_pipe,
                                           param_distributions=param_grid,
                                           cv=5,
                                           n_jobs=-1,
                                           n_iter=20,
                                           return_train_score=True)
        
        random_search.fit(self.X, self.y)
        self.model = random_search.best_estimator_

        results = pd.DataFrame(random_search.cv_results_).set_index("rank_test_score").sort_index()
        results.reset_index(inplace=True)
        results = results.rename(columns = {'index':'Ranked Test Scores'})
        return results

    def get_model(self):
        """
        Returns the optimized model, if present

        Parameters
        ----------

        Returns
        ----------
        sklearn Classifier

        """
        if self.model == None:
            print("Sorry, No optimized model available, please optimize_model() first")
        else:
            return self.model

