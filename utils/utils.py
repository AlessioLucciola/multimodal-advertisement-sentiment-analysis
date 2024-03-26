import pickle
from config import USE_DML, USE_MPS, PATH_TO_SAVE_RESULTS
import random
import numpy as np
import torch
import json
import os

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def select_device():
    if USE_DML:
        import torch_directml
        device = torch_directml.device()
    elif USE_MPS:
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)
    return device

def save_configurations(data_name, configurations):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'configurations.json'
    with open(results_file_path, 'w') as json_file:
        json.dump(configurations, json_file, indent=2)


def save_results(data_name, results, test=False):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/results/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'test_results.json' if test else path + 'tr_val_results.json'
    if os.path.exists(results_file_path):
        final_results = None
        with open(results_file_path, 'r') as json_file:
            final_results = json.load(json_file)
        final_results.append(results)
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)
    else:
        final_results = [results]
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)


def save_model(data_name, model, epoch=None, is_best=False):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/models/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if is_best:
        torch.save(model.state_dict(), f'{path}/mi_project_best.pt')
    else:
        torch.save(model.state_dict(),
                   f'{path}/mi_project_{epoch+1}.pt')
        
def save_scaler(data_name, scaler):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(path + 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def upload_scaler(data_name):
    path = PATH_TO_SAVE_RESULTS + f"/{data_name}/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(path + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler