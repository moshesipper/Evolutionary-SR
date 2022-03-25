# Binary and Multinomial Classification through Evolutionary Symbolic Regression
# copyright 2022 moshe sipper
# www.moshesipper.com

# Utils
from string import ascii_lowercase
from random import choices#, randint
from sys import stdin, exit 
from os import makedirs
from os.path import exists
from pandas import read_csv
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
# from copy import deepcopy 
from scipy.special import expit
import optuna

# ML
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml, make_classification
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from pmlb import fetch_data, classification_dataset_names
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# import optuna.integration.lightgbm as lgb
# GP
from gp import GPClassify, f_div, f_sqrt, f_log
from gplearn.genetic import SymbolicClassifier
from cartesian import Primitive
from cartesian import Symbolic

EASYDS = { 'cancer': load_breast_cancer, 'iris': load_iris, 'wine': load_wine, 'digits': load_digits }
N_TRIALS = 100 # for optuna

def rndstr(n): return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def save_params(fname, dsname, version, n_replicates, n_samples, n_features, n_classes):
    fprint(fname, f' dsname: {dsname}\n version: {version}\n n_samples: {n_samples}\n n_features: {n_features}\n n_classes: {n_classes}\n n_replicates: {n_replicates} \n N_TRIALS: {N_TRIALS} \n')

def get_args():  
    parser = ArgumentParser()
    parser.add_argument('-resdir', dest='resdir', type=str, action='store', help='directory where results are placed')
    parser.add_argument('-dsname', dest='dsname', type=str, action='store', help='dataset name')
    parser.add_argument('-nrep', dest='n_replicates', type=int, action='store', help='number of replicate runs')
    args = parser.parse_args()
    if None in [getattr(args, arg) for arg in vars(args)]:
        parser.print_help()
        exit()
    resdir, dsname, n_replicates = args.resdir+'/', args.dsname, args.n_replicates
    return resdir, dsname, n_replicates

def get_dataset(dsname):
    version, openml = -1, False
    if dsname ==  'clftest':
        X, y = make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
    elif dsname in EASYDS.keys():
        X, y = EASYDS[dsname](return_X_y=True)
    elif dsname in classification_dataset_names: # PMLB datasets
        X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='../datasets/pmlbclf')
    else:
        try: # dataset from openml? 
            data = fetch_openml(data_id=int(dsname), cache=True, data_home='../datasets/scikit_learn_data')
            X, y = data['data'], data['target']
            dsname = data['details']['name']
            version = data['details']['version']
            openml = True
        except:
            try: # a csv file in datasets folder?
                data = read_csv('../datasets/' + dsname + '.csv', sep=',')
                array = data.values
                X, y = array[:,0:-1], array[:,-1] # target is last col
                # X, y = array[:,1:], array[:,0] # target is 1st col
            except Exception as e: # give up
                print('oops, looks like there is no such dataset: ' + dsname)
                exit(e)
              
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    y = LabelEncoder().fit_transform(y) # Encode target labels with value between 0 and n_classes-1
    
    return X, y, n_samples, n_features, n_classes, dsname, version, openml    

class GPLearnClf(BaseEstimator): # n_classes symbolic classifiers, each one-vs-all, based on SymbolicClassifier of gplearn
    def __init__(self, n_pop=-1, n_gens=-1, n_pars=0.001, n_tour=5):
        self.n_pop = n_pop
        self.n_gens = n_gens
        self.n_pars = n_pars
        self.n_tour = n_tour
        self.clfs = []

    def fit(self, X, y): 
        n_classes = len(np.unique(y))
        yhot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
        for cl in range(n_classes):
            self.clfs.append(SymbolicClassifier(population_size=self.n_pop, generations=self.n_gens, tournament_size=self.n_tour, parsimony_coefficient=self.n_pars))
            self.clfs[cl].fit(X, yhot[:,cl])

    def predict(self, X):
        preds = [clf.predict_proba(X)[:,1] for clf in self.clfs]
        return np.argmax(preds, axis=0)
# end class

def log_loss_sig(y_true, y_pred):
    return(log_loss(y_true, expit(y_pred))) # expit == sigmoid

class CartesianClf(BaseEstimator): # n_classes symbolic classifiers, each one-vs-all, based on SymbolicRegression of fastsr
    def __init__(self, n_rows=-1, n_columns=-1, maxiter=-1):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.maxiter = maxiter
        self.clfs = []
        self.primitives = [Primitive("add", np.add, 2), Primitive("sub", np.subtract, 2), Primitive("mul", np.multiply, 2), Primitive("div", f_div, 2), Primitive("abs", np.abs, 1), Primitive("neg", np.negative, 1), Primitive("min", np.minimum, 2), Primitive("max", np.maximum, 2), Primitive("sqrt", f_sqrt, 1), Primitive("log", f_log, 1)]

    def fit(self, X, y): 
        n_classes = len(np.unique(y))
        yhot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
        for cl in range(n_classes):
            self.clfs.append(Symbolic(operators=self.primitives, n_rows=self.n_rows, n_columns=self.n_columns, maxiter=self.maxiter, metric=log_loss_sig))
            self.clfs[cl].fit(X, yhot[:,cl])

    def predict(self, X):
        preds = [clf.predict(X) for clf in self.clfs]
        return np.argmax(preds, axis=0)
# end class

class ClaSyCo(BaseEstimator):
    def __init__(self, n_pop=-1, n_gens=-1, n_pars=0.001, n_tour=5):
        self.n_pop = n_pop
        self.n_gens = n_gens
        self.n_pars = n_pars
        self.n_tour = n_tour
        self.clf = None

    def fit(self, X, y): 
        self.clf = GPClassify(n_pop=self.n_pop, n_gens=self.n_gens, n_pars=self.n_pars, n_tour=self.n_tour)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
# end class

# Algorithms = [ClaSyCo, GPLearnClf, CartesianClf, RandomForestClassifier, AdaBoostClassifier, MLPClassifier, XGBClassifier]
Algorithms = [ClaSyCo, GPLearnClf, CartesianClf, XGBClassifier, LGBMClassifier, MLPClassifier]
# Algorithms = [LGBMClassifier]

def create_model(trial):
    clf = trial.suggest_categorical('classifier', [a.__name__ for a in Algorithms])       
    if clf == 'ClaSyCo':
        ClaSyCo_n_pop = trial.suggest_int('ClaSyCo_n_pop', 20, 200, log=False)
        ClaSyCo_n_gens = trial.suggest_int('ClaSyCo_n_gens', 20, 200, log=False)
        model = ClaSyCo(n_pop=ClaSyCo_n_pop, n_gens=ClaSyCo_n_gens)
    elif clf == 'GPLearnClf':
        GPLearnClf_n_pop = trial.suggest_int('GPLearnClf_n_pop', 20, 200, log=False)
        GPLearnClf_n_gens = trial.suggest_int('GPLearnClf_n_gens', 20, 200, log=False)
        model = GPLearnClf(n_pop=GPLearnClf_n_pop, n_gens=GPLearnClf_n_gens)
    elif clf == 'CartesianClf':
        CartesianClf_n_rows = trial.suggest_int('CartesianClf_n_rows', 1, 10, log=False)
        CartesianClf_n_columns = trial.suggest_int('CartesianClf_n_columns', 1, 10, log=False)
        CartesianClf_maxiter = trial.suggest_int('CartesianClf_maxiter', 10, 1000, log=True)
        model = CartesianClf(n_rows=CartesianClf_n_rows, n_columns=CartesianClf_n_columns, maxiter=CartesianClf_maxiter)
    elif clf == 'RandomForestClassifier':
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 10, 1000, log=True)
        rf_min_weight_fraction_leaf = trial.suggest_float('rf_min_weight_fraction_leaf', 0, 0.5, log=False)
        rf_max_features = trial.suggest_categorical('rf_max_features', ['auto','sqrt','log2'])
        model = RandomForestClassifier(n_estimators=rf_n_estimators, min_weight_fraction_leaf=rf_min_weight_fraction_leaf,\
                                       max_features=rf_max_features)
    elif clf == 'AdaBoostClassifier':
        ada_n_estimators = trial.suggest_int('ada_n_estimators', 10, 1000, log=True)
        ada_learning_rate = trial.suggest_float('ada_lr', 0.01, 10, log=True)
        model = AdaBoostClassifier(n_estimators=ada_n_estimators, learning_rate=ada_learning_rate)
    elif clf == 'MLPClassifier':
        mlp_activation = trial.suggest_categorical('mlp_activation', ['identity', 'logistic', 'tanh', 'relu'])
        mlp_solver = trial.suggest_categorical('mlp_solver', ['lbfgs', 'sgd', 'adam'])
        mlp_learning_rate = trial.suggest_categorical('mlp_lr', ['constant', 'invscaling', 'adaptive'])
        model = MLPClassifier(activation=mlp_activation, solver=mlp_solver, learning_rate=mlp_learning_rate, hidden_layer_sizes=(16,16,16,16,16,16,16,16,16,16))
    elif clf == 'XGBClassifier':
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 10, 1000, log=True)
        xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=False)
        xgb_gamma = trial.suggest_float('xgb_gamma', 0, 0.4, log=False)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 4, 6, log=False)
        xgb_subsample = trial.suggest_float('xgb_subsample', 0.5, 1, log=False)
        model = XGBClassifier(n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate, gamma=xgb_gamma, max_depth=xgb_max_depth, subsample=xgb_subsample)
    elif clf == 'LGBMClassifier':
        lgbm_n_estimators = trial.suggest_int('lgbm_n_estimators', 10, 1000, log=True)
        lgbm_learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.2, log=False)
        lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 4, 6, log=False)
        lgbm_bagging_fraction = trial.suggest_float('lgbm_bagging_fraction', 0.5, 0.95, log=False)
        lgbm_bagging_freq = trial.suggest_int('lgbm_bagging_freq', 1, 10, log=False)
        model = LGBMClassifier(n_estimators=lgbm_n_estimators, learning_rate=lgbm_learning_rate, max_depth=lgbm_max_depth, bagging_fraction=lgbm_bagging_fraction, bagging_freq=lgbm_bagging_freq)
    else:
        exit('error: unknown regressor')
    return model

class Objective(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, trial):
        model = create_model(trial)    
        model.fit(self.X_train, self.y_train)
        return balanced_accuracy_score(self.y_test, model.predict(self.X_test))
# end class Objective

# main 
def main():
    resdir, dsname_id, n_replicates = get_args()
    X, y, n_samples, n_features, n_classes, dsname, version, openml = get_dataset(dsname_id)
    print_ds = f'{dsname} ({dsname_id})' if openml else f'{dsname}' # openml datasets are given as ints and get_dataset converts to string, print both
    if not exists(resdir): 
        makedirs(resdir)
    fname = resdir + dsname_id + '_' + rndstr(6) + '.txt'
    save_params(fname, print_ds, version, n_replicates, n_samples, n_features, n_classes)
       
    for rep in range(1, n_replicates+1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # X_train = StandardScaler().fit_transform(X_train) # Scaled data has zero mean and unit variance
        # X_test = StandardScaler().fit_transform(X_test)
        sc = StandardScaler() 
        X_train = sc.fit_transform(X_train) # scaled data has mean 0 and variance 1 (only over training set)
        X_test = sc.transform(X_test) # use same scaler as one fitted to training data
        
        objective = Objective(X_train, X_test, y_train, y_test)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        best_clf = str(study.best_trial.params)#study.best_trial.params['classifier']
        best_score = round(study.best_trial.value,3)
        fprint(fname, f'\n {print_ds} & {n_samples} & {n_features} & {n_classes} & {best_score} & {best_clf}')
 
    fprint(fname, '\n')         
    
##############        
if __name__== "__main__":
  main()

