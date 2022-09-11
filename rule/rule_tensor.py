import numpy as np
import matplotlib.pyplot as plt

class SmallVarianceRule():
    def __init__(self, base_trial, min_threshold=0.0001):
        super().__init__()
        self.min_threshold = float(min_threshold)
        self.base_trial = base_trial

    def invoke_at_step(self, step):
        step_vars = list()
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            var = t.reduction_value(step, "variance")
            is_small = False
            if var < self.min_threshold:
                is_small = True
            step_var = (tname, var, is_small)
            step_vars.append(step_var)
        return step_vars
    
    def work(self):
        steps = self.base_trial.steps()
        for step in steps:
            print("step ", step, ":")
            step_vars = self.invoke_at_step(step)
            layer_names = list(step_vars[i][0] for i in range(4))
            vars = list(step_vars[i][1] for i in range(4))
            plt.bar(layer_names, vars)
            plt.show()      
            for step_var in step_vars:
                if step_var[2] == True:
                    print(step_var[0], ": Variance of values is too small")


class ValuesUnchangedRule():
    def __init__(self, base_trial, rtol=1e-01, atol=1e-01):
        super().__init__()
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.base_trial = base_trial

    def invoke_at_step(self, last_step, cur_step):
        step_tensors = list()
        for tname in self.base_trial.tensor_names(collection="weights"):
            last_t = self.base_trial.tensor(tname).value(last_step)
            cur_t = self.base_trial.tensor(tname).value(cur_step)
            is_unchanged = np.allclose(
                last_t, cur_t, 
                rtol=self.rtol, 
                atol=self.atol,
                equal_nan=False)
            step_tensor = (tname, is_unchanged)
            step_tensors.append(step_tensor)
        return step_tensors
    
    def work(self):
        steps = self.base_trial.steps()
        last_step = steps[0]
        for step in steps: 
            print("step ", step, ":")
            step_tensors = self.invoke_at_step(last_step, step) 
            for step_var in step_tensors:
                if step_var[1] == True and step != steps[0]:
                    print(step_var[0], ": Tensors were unchanged")
                print(step_var[0], ": Tensors changed properly")
            last_step = step