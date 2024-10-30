import sys

from source.nn import CNN, TransferLearning, cross_validation
from source import data_loader

def Test_and_Train(config):

    '''execute Transfer Learning + predict test data'''

    # train_features ... brightest picture: 1370


    (train_features__HF, train_target__HF,
     test_features, test_target) = data_loader.load_TrainData(config, config["file__train__HF"], "HF")
    train_features__LF, train_target__LF = data_loader.load_TrainData(config, config["file__train__LF"], "LF")

    #val_features, val_target = data_loader.load_TestData(config, config["file__test"], index=4351)
    #test_features, test_target = data_loader.load_TestData(config, config["file__test"], index=4342)


    CNN_model = CNN(config)
    #CNN_model.plot_stats__nn()


    '''training'''

    TransferLearning(CNN_model, config,
                        train_features__HF, train_target__HF,
                        train_features__LF, train_target__LF)

    CNN_model.model.save(data_loader.resource_path(config["checkpoint_path"]))

    '''plot+analyse'''

    CNN_model.load_model(config)
    CNN_model.predicts(test_features)
    CNN_model.analyse(train_features__HF, train_target__HF, test_features, test_target)






def PerformanceCheck(config):

    features__HF, target__HF = data_loader.load_TrainData(config, config["file__train__HF"])
    features__LF, target__LF = data_loader.load_TrainData(config, config["file__train__LF"])

    average_acc, average_loss, file__maxAccuracy = cross_validation(config, # K-Fold Cross-Validation
                                                                    features__HF, target__HF,
                                                                    features__LF, target__LF)

    print("Average (Test)-Accuracy:", average_acc)
    print("Average (Test)-Loss:", average_loss)
    print("Model with optimal (Test-)Accuracy:", file__maxAccuracy)


