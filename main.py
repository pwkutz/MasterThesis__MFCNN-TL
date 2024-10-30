# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import os
from source import nn
from source.nn import CNN
from source import data_loader
from source.Modes import Test_and_Train, PerformanceCheck



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    config = data_loader.load_yaml(os.path.abspath(".") + "/" + "config.yaml")
    nn.safety_check(config)

    if config["mode"]["TestAndTrain"]: Test_and_Train(config)
    elif config["mode"]["PerformanceCheck"]: PerformanceCheck(config)







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
