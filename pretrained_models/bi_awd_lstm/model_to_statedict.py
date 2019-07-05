import torch
from torch.nn import Parameter

'''
This file is stritly for use of converting a saved model to a state dictionary, 
as saved models are hard to work with between versions and refactors.
'''

def convert_model_to_statedict(model_path="pretrained_models/bi_awd_lstm/elmo_parameters.pt"):
    model = None
    with open(model_path, 'rb') as f:
        model, _, _ = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    return model

if __name__ == "__main__":
    model = convert_model_to_statedict("pretrained_models/bi_awd_lstm/elmo_parameters.pt")
    torch.save(model.state_dict(), "pretrained_models/bi_awd_lstm/elmo_parameters_statedict.pt")