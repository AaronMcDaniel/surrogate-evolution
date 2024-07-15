
import toml


class Surrogate():
    def __init__(self, config_dir):
        # Begin by loading config attributes
        configs = toml.load(config_dir)
        surrogate_config = configs["surrogate"]
        self.models = surrogate_config["models"]
        self.trust_calc_strategy = surrogate_config["trust_calc_strategy"]
        self.trust_calc_ratio = surrogate_config["trust_calc_ratio"]
        
        self.trusts = [0 for _ in self.models]
    
    # The calc_pool is a list of deap individuals with calculated fitnesses. The model infers the metrics and 
    # we see the intersection in selections
    def calc_trust(self, model_name, calc_pool):
        if model_name.lower() not in self.models:
            raise ValueError(f'{model_name} provided is not in list of surrogate models')
        
        # create copy of calc_pool
        
        # convert deap individuals to string and call codec.encode_surrogate
        
        # get inferences on copy of calc_pool and assign fitness to copy
        
        # run trust-calc strategy to select trust_calc_ratio-based number of individuals for both calc_pool and its copy
        
        # check intersection of selected individuals and return
        pass
    
    # This function converts string representations of genomes from a file like out.csv into deap individuals
    # with fitness that can be used to either train on or calculate trust with
    # parameter generations tells us what generations to get individuals from. Will use all individuals in a file if unspecified
    def get_individuals_from_file(filepath, generations=None): 
        pass
    
    
    def get_best_model(self):
        return self.models[max(enumerate(self.trusts), key=lambda x: x[1])[0]]
        