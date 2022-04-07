import torch


class EarlyStopper(object):
    
    def __init__(self, num_trials, direction, save_path):
        self.num_trials = num_trials       
        self.save_path = save_path
        self.direction = direction
        self.trial_counter = 0 
        self.best_metric = 0 if direction == 'minimize' else 1
        
    def is_continuable(self, model, metric):
        if self.directoin == 'minimize':
            if metric < self.best_metric:
                self.best_accuracy = metric
                self.trial_counter = 0
                torch.save(model, self.save_path)
                return True
            elif self.trial_counter + 1 < self.num_trials:
                self.trial_counter += 1
                return True
            else:
                return False
        else:
            if metric > self.best_metric:
                self.best_metric = metric
                self.trial_counter = 0
                torch.save(model, self.save_path)
                return True
            elif self.trial_counter + 1 < self.num_trials:
                self.trial_counter += 1
                return True
            else:
                return False