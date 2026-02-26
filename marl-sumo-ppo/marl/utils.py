import numpy as np

def flatten_dict_observations(obs_dict, agents):
    return np.concatenate([obs_dict[a] for a in agents]).astype(np.float32)