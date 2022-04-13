import torch


class EarlyStopper(object):
    
    def __init__(self, num_trials, direction, save_path=None):
        self.num_trials = num_trials       
        self.save_path = save_path
        self.direction = direction
        self.trial_counter = 0 
        self.best_metric = 1 if direction == 'minimize' else 0
        
    def is_continuable(self, model, metric):
        if self.direction == 'minimize':
            if metric < self.best_metric:
                self.best_accuracy = metric
                self.trial_counter = 0
                if self.save_path: torch.save(model, self.save_path)
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
                if self.save_path: torch.save(model, self.save_path)
                return True
            elif self.trial_counter + 1 < self.num_trials:
                self.trial_counter += 1
                return True
            else:
                return False