import importlib
from ipdb import set_trace as st


# find the dataset definition by name, for example dtu_yao (dtu_yao.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    # st()
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")
