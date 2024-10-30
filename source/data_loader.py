import sys
import os
import pandas as pd
import numpy as np
import ruamel.yaml
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from source import nn



def resource_path(relative_path):
    '''
    return total path for a given relative path
    total path is "pyinstaller conform"
    '''

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def load_yaml(file_name: str):  # -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """



    # Use this to load your config file
    file_name = resource_path(file_name)



    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None

def show_image(image):

    plt.imshow(image, interpolation='nearest')
    plt.show()

def remove_contentless_images(features, target):

    '''remove all images (LF/HF) which do not possess any content.
    Contentless images are image those pixels possess all the same value (e.g., 0)'''

    index__NonZero = [x for x in range(len(features)) if len(features.iloc[x].unique())>1]
    features = features.iloc[index__NonZero].reset_index(drop=True)
    target = target.iloc[index__NonZero].reset_index(drop=True)

    return features, target

def split__train_and_test(features, target):

    '''
    split existing HF dataset for training into train and test dataset
    Original test dataset consists out to 100 % out of content-free images (not usable)
    '''

    num__train = int(np.round(len(features) * 0.9))
    num__test = int(len(features) - num__train)
    all_samples = list(range(len(features)))

    index__test = list(range(0, len(all_samples), int(len(all_samples)/num__test)))

    index__train = [x for x in all_samples if x not in index__test]

    feat__train = features[index__train]
    target__train = target.iloc[index__train]

    feat__test = features[index__test]
    target__test = target.iloc[index__test]

    return feat__train, target__train, feat__test, target__test

def preprocessing(window_size, features, target, split=False):

    '''smooth graphs (moving average) + normalize input data + split HF into train and test data'''

    features, target = remove_contentless_images(features, target)

    features, target = apply__moving_average(window_size, features, target)
    features = features.T # MinMaxScaler normalizes columns-wise only. But the flattened images are row-wise inside df
    features, target, scaler__target = normalize_data(features, target) # normalize each image separately

    features = pd.DataFrame(features).T # re-arrange image DataFrame to old order where images are sorted row-wise.
    features = nn.transform_to_scans(features)
    #show_image(features[0])
    target = map(lambda x: x[0], target) # turn target from Numpy Array back to Pandas Series
    target = pd.Series(target)

    if split: # HF
        feat__train, target__train, feat__test, target__test = split__train_and_test(features, target)
        return feat__train, target__train, feat__test, target__test, scaler__target
    else: # LF
        return features, target, scaler__target

def moving_average(window_size, Pandas_Series):

    '''apply a moving average to a given series of values. Goal: smoothing + denoising the series'''

    Pandas_Series = Pandas_Series.rolling(window=window_size, center=True).mean()
    return Pandas_Series.dropna()

def apply__moving_average(window_size, feat, target):

    '''
    apply moving average to the train, val and test dataset
    COMMENTS: decomment to plot effect of Moving Average
    '''

    target = moving_average(window_size, target)
    common_idx = feat.index.intersection(target.index)
    feat = feat.loc[common_idx]

    target = target.reset_index(drop=True)
    feat = feat.reset_index(drop=True)

    return feat, target


def load_TrainData(config, file, fidelity=None):

    ''' load Training Data as DataFrame into program '''

    train = pd.read_csv(resource_path(file), header=None)
    features = train.iloc[:, :4096]
    outputs = train.iloc[:, 4096:]
    target = outputs.iloc[:, 0]
    if fidelity == "HF":
        (train_features, train_target,
         feat__test, target__test,
         scaler__train_target) = preprocessing(config["window_size"]["train"], features, target, True)
        return train_features, train_target, feat__test, target__test

    else:
        (train_features, train_target,
        scaler__train_target) = preprocessing(config["window_size"]["train"], features, target)
        return train_features, train_target


def load_TestData(config, file, index):

    '''load Test Data as DataFrame into program'''

    test = pd.read_csv(resource_path(file), header=None)

    run_indx = test.iloc[:,-1] == index
    test_features = test.iloc[:, :4096][run_indx]
    test_outputs=test.iloc[:, 4096:][run_indx]
    test_target = test_outputs.iloc[:, 0]

    test_features, test_target, scaler__test_target = preprocessing(config["window_size"]["test"], test_features, test_target)

    return test_features, test_target

def load_CommonDataset(config):

    '''
    HF + LF features and targets
    '''

    train_features, train_target = load_TrainData(config["file__train__HF"]) # load whole Train Dataset
    test_features, test_target = load_TrainData(config["file__test"]) # load whole Test Dataset
    features__LF, target__LF = load_TrainData(config["file__train__LF"])

    features__HF = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)  # retrieve common Dataset
    target__HF = pd.concat([train_target, test_target], axis=0).reset_index(drop=True)

    return features__HF, target__HF, features__LF, target__LF

def normalize(df):

    '''
    Normalize Pandas object column-wise
    Each column gets normalized separately
    '''

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    return scaled, scaler

def de_normalize(df, scaler):
    unscaled = scaler.inverse_transform(df)
    return unscaled

def normalize_data(feat, target):

    feat, _ = normalize(feat)
    target, scaler__target = normalize(pd.DataFrame(target))

    return feat, target, scaler__target
