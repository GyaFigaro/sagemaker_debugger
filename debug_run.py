from distutils.command.config import config
from types import NoneType
import smdebug.pytorch as smd
from rule.rule_tensor import Rule_Tensor
from rule.rule_functions import Rule_ActivationFunctions
from rule.rule_loss import Rule_Loss
import json, os

config_path = "./rule_config.json"
accuracy_path = './accuracy.npy'

class Rule_Config():
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def deploy(self):
        return 

config = Rule_Config(config_path=config_path)
trial = smd.create_trial(path="./tmp/testing/demo")
steps = trial.steps(mode=smd.modes.TRAIN)
    
rule_tensor = Rule_Tensor(base_trial=trial, steps=steps)
rule_tensor.all_values_zero()
rule_tensor.values_unchanged()

rule_func = Rule_ActivationFunctions(base_trial=trial, steps=steps)
rule_func.sigmoid_saturation()
rule_func.tanh_saturation()
rule_func.dying_relu()

# rule_loss = Rule_Loss(base_trial=trial)
# print(rule_loss.Loss_Not_Decreasing(increase_threshold_percent=100))
# print(rule_loss.Overfitting(20))
# print(rule_loss.Underfitting(0, accuracy_path, 90))
