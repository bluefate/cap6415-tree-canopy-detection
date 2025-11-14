#%% md
# # Notebook: 0 Tester Notebook
#%%
from src.utils.tester import test_print


test_print()
#%%
from src.utils.config import Config


config = Config.load()
config.show()

#%%
from src.utils.helpers import init_notebook


init_notebook(config.train.seed)

#%%
def print_versions():
    from src.utils.helpers import p
    import sys

    import numpy as np
    import pandas as pd
    import torch


    p("Python", sys.version)
    p("Numpy", np.__version__)
    p("Panda", pd.__version__)
    p("Torch", torch.__version__)


#%%
print_versions()