from distutils.command.config import config
from types import NoneType
import smdebug.pytorch as smd
from rule.rule_tensor import Rule_Tensor
from rule.rule_functions import Rule_ActivationFunctions

import os
import json

config_path = "./rule_config.json"

class Rule_Config():
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def deploy(self):
        return 

def debug(epoch):
    
    # config = Rule_Config(config_path=config_path)
    trial = smd.create_trial(path="./tmp/testing/demo")

    if not os.path.exists("./debug_info"):
        os.makedirs("./debug_info")
    epo_info = {'epoch_num': epoch}
    with open("./debug_info/epoch_info.json","w") as f:
        json.dump(epo_info, f)
        print("epoch info loads successfully")

    rule_tensor = Rule_Tensor(base_trial=trial)
    rule_tensor.all_values_zero()
    rule_tensor.values_unchanged()

    rule_func = Rule_ActivationFunctions(base_trial=trial)
    rule_func.sigmoid_saturation()
    rule_func.tanh_saturation()
    rule_func.dying_relu()



    
