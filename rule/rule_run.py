from distutils.command.config import config
from types import NoneType
import smdebug.pytorch as smd
from rule.rule_tensor import Rule_Tensor
from rule.rule_functions import Rule_ActivationFunctions
from rule.rule_datasets import Rule_Datasets
from rule.rule_loss import Rule_Loss
from rule.rule_gradients import Rule_Gradients
from rule.rule_weights import Rule_Weights

import os
import json

config_path = "./rule_config.json"
accuracy_path = './accuracy.npy'

rule_dataset = Rule_Datasets()

class Rule_Config():
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def deploy(self):
        return 

def epoch_debug(epoch, batch_num):
    
    trial = smd.create_trial(path="./tmp/testing/demo")
    steps = trial.steps(mode=smd.modes.TRAIN)[(int)(938 * epoch / 100) + 1:]
    # config = Rule_Config(config_path=config_path)
    
    if not os.path.exists("./debug_info"):
        os.makedirs("./debug_info")

    epo_info = {'epoch_num': epoch - 1,
                'overfitting': False,
                'underfitting': False,
                'loss_not_decrease': False,
                'poor_initialization': False,
                'update_small': False,
                'vanishing_gradient': False,
                'exploding_gradient': False,
                'all_values_zero': False,
                'tensors_unchanged': False,
                'dead_relu': False,
                'tanh/sigmoid_saturation': False
                }
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epo_info, f)
        print("epoch info loads successfully")



    rule_loss = Rule_Loss(base_trial=trial)
    rule_gradients = Rule_Gradients(base_trial=trial)
    rule_weight = Rule_Weights(base_trial=trial)
    rule_tensor = Rule_Tensor(base_trial=trial, steps=steps)
    rule_func = Rule_ActivationFunctions(base_trial=trial, steps=steps)

    print(rule_loss.Loss_Not_Decreasing(increase_threshold_percent=100))
    print(rule_loss.Overfitting(0))
    print(rule_loss.Underfitting(0, accuracy_path, 90))
    
    rule_gradients.gradients()

    rule_weight.poor_initialization()
    rule_weight.update_too_small()

    rule_tensor.all_values_zero()
    rule_tensor.values_unchanged()
    
    rule_func.sigmoid_saturation()
    rule_func.tanh_saturation()
    rule_func.dying_relu()

    epo_info['epoch_num'] = epoch
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epo_info, f)
        print("epoch info loads successfully")

def classfier_debug(category_no, labels, predictions):
    if not os.path.exists("./debug_info"):
        os.makedirs("./debug_info")
    trial = smd.create_trial(path="./tmp/testing/demo")
    rule_loss = Rule_Loss(base_trial=trial)
    return rule_loss.Classifier_Confusion(category_no, labels, predictions)

def dataset_debug(mode, train_loader):
    if not os.path.exists("./debug_info"):
        os.makedirs("./debug_info")
    if mode == 'balance':
        return rule_dataset.input_balance(train_loader)
    if mode == 'normalize':
        return rule_dataset.Not_Normalized_Data(train_loader)




    
