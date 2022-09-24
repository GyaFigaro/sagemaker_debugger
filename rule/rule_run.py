from distutils.command.config import config
from types import NoneType
import smdebug.pytorch as smd
from rule.rule12to15 import SigmondSaturationRule   
from rule.rule_tensor import Rule_Tensor

config_path = "./rule_config.json"

class Rule_Config():
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def deploy(self):
        return 

def debug():
    config = Rule_Config(config_path=config_path)
    trial = smd.create_trial(path="./tmp2/testing/demo")
    
    rule_tensor = Rule_Tensor(base_trial=trial)
    rule_tensor.all_values_zero()
    rule_tensor.values_unchanged()
    
