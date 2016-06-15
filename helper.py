import numpy as np
import pandas as pd

import json


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree, ensemble
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import roc_curve, auc


def from_json_to_dict(date, path='/Users/akhil/weather/', prefix='weather_kord_2001_'):
    """
    Takes a string in the format MMDD where M = month, D = day of month.
    Read a json file at "path" + "prefix" + "date".
    Returns the JSON dictionary.

    Parameters
    ----------
    date: A string.

    Optional
    --------
    path: A string.
    prefix: A string.

    Returns
    -------
    A dict.
    """

    with open(path + prefix + date + '.json') as data_file:
        data = json.load(data_file)

    return data


def from_dict_to_visibility(json_data):
    """
    Takes a dictionary and returns a tuple of (Month, Day, Hour, Minute, Visibility).

    Parameters
    ----------
    json_data: A dict.

    Returns
    -------
    A 5-tuple (str, str, str, str, str)
    """

    result = list()

    for obs in json_data['history']['observations']:
        month = obs['date']['mon']
        day = obs['date']['mday']
        hour = obs['date']['hour']
        minute = obs['date']['min']
        visibility = obs['visi']
        result.append((month, day, hour, minute, visibility))

    return result


def collect_365_days(dates):
    """
    Uses from_json_to_dict() and from_dict_to_visiblility() to
    generate a list of tuples of the form
    (Month, Day, Hour, Minute, Visibility)

    Parameters
    ----------
    dates: A list of strings "MMDD"

    Returns
    -------
    A list of 5-tuples (str, str, str, str, str)
    """

    visibilities = list()

    for date in dates:
        data = from_json_to_dict(date)
        visibilities.extend(from_dict_to_visibility(data))

    return visibilities


def from_string_to_numbers(visibilities):
    """
    Takes a list of 5-tuples of strings.
    Convert the strings into integers in the form `mmddHHMM`,
    where `m` is month, `d` is day of month, `H` is hour, and `M` is minute.
    Returns a pandas.DataFrame with two columns "Time" and "Visibility".

    Parameters
    ----------
    visibilities: A list of 5-tuple of strings.

    Returns
    -------
    A pandas.DataFrame
    """

    numbers = list()
    visis = list()

    # tuple is month,day,hour,minute,visibility
    for item in visibilities:
        number = int(item[0] + item[1] + item[2] + item[3])
        visi = float(item[4])
        numbers.append(number)
        visis.append(visi)

    result = pd.DataFrame({'Time': numbers, 'Visibility': visis})

    return result


def combine_time(df):
    """
    Combines "Month", "DayofMonth", and "CRSDepTime" in the form mmddHHMM.
    Creates a new column named "Time".

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A pandas.DataFrame
    """

    time = list()

    for row in df.iterrows():
        number = int("%02d" % (row[1]['Month'],) + "%02d" % (row[1]['DayofMonth'],) + "%04d" % (row[1]['CRSDepTime'],))
        time.append(number)

    result = df.copy()
    result['Time'] = time

    return result


def match_visibility(df_delayed, df_visibility, inplace=False):
    if inplace:
        # we'll make changes right on df_delayed
        result = df_delayed
    else:
        # we don't want to change the original data frame
        result = df_delayed.copy()

    # get the numpy arrays for the flight and visibility times
    flight_time = df_delayed['Time'].values
    vis_time = df_visibility['Time'].values

    # find the index of first visibility time greater than each flight time
    idx = np.searchsorted(vis_time, flight_time)
    # constrain to be valid array index > 0
    idx = np.clip(idx, 1, len(vis_time) - 1)

    # find index of closest visibility time, either idx or idx-1
    # note this comparison will properly handle when a flight time is prior to all visibility readings
    # and when the flight time is after all visibility readings
    # The `<=` is chosen to match the result of the non-vectorized version
    idx = np.where(flight_time - vis_time[idx - 1] <= vis_time[idx] - flight_time,
                   idx - 1, idx)

    result['Visibility'] = df_visibility['Visibility'][idx].values

    return result


def split(df, test_column, test_size, random_state):
    """
    Uses sklearn.train_test_split to split "df" into a testing set and a test set.
    The "test_columns" lists the column that we are trying to predict.
    All columns in "df" except "test_columns" will be used for training.
    The "test_size" should be between 0.0 and 1.0 and represents the proportion of the
    dataset to include in the test split.
    The "random_state" parameter is used in sklearn.train_test_split.

    Parameters
    ----------
    df: A pandas.DataFrame
    test_columns: A list of strings
    test_size: A float
    random_state: A numpy.random.RandomState instance

    Returns
    -------
    A 4-tuple of pandas.DataFrames
    """
    return train_test_split(df.ix[:, df.columns != test_column], df[test_column],
                            test_size=test_size, random_state=random_state)


def fit_decision(X_train, y_train, X_test, random_state):
    """
    Fits Decision Trees.

    Parameters
    ----------
    X: A pandas.DataFrame. Training attributes.
    y: A pandas.DataFrame. Truth labels.

    Returns
    -------
    A numpy array.
    """

    dtc = tree.DecisionTreeClassifier(random_state=random_state)
    dtc.fit(X_train, y_train)

    prediction = dtc.predict(X_test)

    return prediction


def get_cv_indices(df, n_folds, random_state):
    """
    Provides train/test indices to split data in train test sets.
    Split dataset into "n_folds" consecutive folds (no shuffling).

    Paramters
    ---------
    df: A pandas.DataFrame
    n_folds: integer
    random_state: A numpy.random.RandomState instance

    Returns
    -------
    An sklearn.cross_validation.KFold instance.
    """

    result = KFold(len(df), n_folds, random_state=random_state)

    return result


def get_rfc(n_estimators, max_features, random_state):
    """
    A random forest classifier with two adjustable parameters:
    "n_estimators" and "max_features".
    Uses the default sklearn values for the remaining parameters.

    Parameters
    ----------
    n_estimators: An int
    max_features: An int
    random_state: A numpy.random.RandomState instance

    Returns
    -------
    An sklearn.ensemble.forest.RandomForestClassifier
    """

    rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                          random_state=random_state)

    return rfc


def get_proba(clf, X_train, y_train, idx_train, idx_valid):
    """

    Fits the "clf" model on X_train[idx_train] and y_train[idx_train].
    Makes predictions on X_train[idx_valid].

    Parameters
    ----------
    clf: An sklearn classifier instance.
    X_train: A pandas.DataFrame
    y_train: A pandas.DataFrame
    idx_train: A numpy array
    idx_valid: A numpy array

    Returns
    -------
    A two-dimensional numpy array
    """

    clf.fit(X_train.iloc[idx_train], y_train.iloc[idx_train].values.ravel())
    y_pred = clf.predict_proba(X_train.iloc[idx_valid])

    return y_pred


def get_auc(kf, rfc, X_train, y_train):
    """
    Iterates through the cross-validation folds and
    returns the area under the ROC curve for the validation data.

    Parameters
    ----------
    kf: An sklearn.KFold instance
    rfc: An sklearn.RandomForestClassifer instance
    X_train: A pandas.DataFrame
    y_train: A pandas.DataFrame

    Returns
    -------
    A numpy array
    """

    arealist = list()

    for idx_train, idx_valid in kf:
        y_pred = get_proba(rfc, X_train, y_train, idx_train, idx_valid)
        fpr, tpr, _ = roc_curve(y_train.iloc[idx_valid], y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        arealist.append(roc_auc)

    result = np.array(arealist)
    return result


def optimize_max_features(X_train, y_train, cv_random_state, clf_random_state, n_folds=4, n_trees=20):
    """

    Parameters
    ----------
    X_train: A pandas.DataFrame
    y_train: A pandas.DataFrame
    cv_random_state: A RandomState instance for get_cv_indices()
    clf_random_state: A RandomState instance for get_auc()

    Optional
    --------
    n_folds: An int. 4 by default.
    n_trees: An int. 20 by default.
    """

    result = list()

    for i in range(1, 8):
        kf = get_cv_indices(X_train, n_folds, cv_random_state)
        rfc = get_rfc(n_trees, i, clf_random_state)
        auc_value = get_auc(kf, rfc, X_train, y_train).mean()
        result.append((i, auc_value))

    return result


def get_final_rfc(X_train, y_train, X_test, max_features, random_state, n_trees=100):
    """
    Trains a Random Forest classifier on the entire training set
    using the optimized "max_features".
    Predicts

    Parameters
    ----------
    X_train: A pandas.DataFrame
    y_train: A pandas.DataFrame
    X_test: A pandas.DataFrame
    max_features: An int
    random_state: A RandomState instance

    Optional
    --------
    n_trees: An int. 100 by default

    Returns
    -------
    A two-dimensional numpy array
    """

    rfc = get_rfc(n_trees, max_features, random_state)
    rfc.fit(X_train, y_train.values.ravel())
    y_pred = rfc.predict_proba(X_test)

    return y_pred


def plot_roc_curve(y_test, y_pred):
    """
    Plots ROC curve with FPR on the x-axis and TPR on the y-axis.
    Displays AUC ROC in the legend.

    Paramters
    ---------
    y_test: A pandas.DataFrame
    y_pred: A two dimensional array from get_final_rfc()

    Returns
    -------
    A matplotlib.Axes instance
    """

    sns.set(style="darkgrid", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label='Area = ' + str(roc_auc))

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='best')

    sns.despine(offset=0, trim=True)

    return ax