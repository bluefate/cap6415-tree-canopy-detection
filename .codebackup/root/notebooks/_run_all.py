# %% [markdown]
# # Notebook: Run All
#

# %%
import glob
import os
import sys

import nbformat
from nbclient import NotebookClient


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))
from src.utils.helpers import p, t, c


t("notebooks")

root = os.path.dirname(os.getcwd())
source_folder = os.path.join(root, "notebooks")
p("source_folder", source_folder)
p()
notebooks = glob.glob(os.path.join(source_folder, "*.ipynb"))
notebooks = [nb for nb in notebooks if os.path.basename(nb) != "_run_all.ipynb"]

for nb in notebooks:
    print(os.path.basename(nb))



# %%
failed_notebooks = []

for nb_path in notebooks:
    p()
    try:
        t(f"Executing: {nb_path}")
        nb = nbformat.read(nb_path, as_version = 4)
        client = NotebookClient(
                nb,
                timeout = 900,
                kernel_name = "python3"
        )
        client.execute()
        #save notebook
        nbformat.write(nb, nb_path)
        p(f"Finished & saved: {nb_path}", color1 = c.BLUE)
    except Exception as e:
        p(f"Error while executing {nb_path}, {e}", color1 = c.RED, color2 = c.BLACK)
        failed_notebooks.append(nb_path)

t("\nExecution completed.")

if failed_notebooks:
    p("\n\nNotebooks that failed:", color1 = c.MAGENTA)
    for nb_path in failed_notebooks:
        p("", nb_path)

