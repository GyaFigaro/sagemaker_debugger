from distutils.command.config import config
from types import NoneType
import smdebug.pytorch as smd
from rule.rule_tensor import Rule_Tensor
from rule.rule_functions import Rule_ActivationFunctions
from rule.rule_loss import Rule_Loss

config_path = "./rule_config.json"

class Rule_Config():
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def deploy(self):
        return 


config = Rule_Config(config_path=config_path)
trial = smd.create_trial(path="./tmp/testing/demo")
    
# rule_tensor = Rule_Tensor(base_trial=trial)
# rule_tensor.all_values_zero()
# rule_tensor.values_unchanged()

# rule_func = Rule_ActivationFunctions(base_trial=trial)
# rule_func.sigmoid_saturation()
# rule_func.tanh_saturation()
# rule_func.dying_relu()