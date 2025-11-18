# %% [markdown]
# # Notebook: 02 Class Annotation Exploration
# ### Purpose: This notebook provides utilities for exploring masks, overlays, classes, and bounding boxes

# %% [markdown]
# #### Config and Setup

# %%
from src.data.annotations import get_unique_classes, load_json_annotations
from src.exploration.class_explorer import *
from src.utils.config import Config
from utils.helpers import init_notebook, p


config = Config.load()
init_notebook(config.train.seed)



# %%

annotations_path = config.paths.annotations

entries = load_json_annotations(annotations_path)
unique_classes = get_unique_classes(entries)

p("Loaded annotation entries", len(entries))
p("Unique annotation classes", unique_classes)


# %%
entry = entries[config.train.seed]
p("Classes found in entry", count_classes(entry))
# p("Number of classes found", class_distribution(entry))

# %%


show_single_class(entry, config.paths.train_images, "individual_tree")
show_single_class(entry, config.paths.train_images, "group_of_trees")
show_all_classes(entry, config.paths.train_images)
show_per_class(entry, config.paths.train_images)
show_overlay_all(entry, config.paths.train_images)
show_overlay_by_class(entry, config.paths.train_images)



# %% [markdown]
# #### Explore Masks Per Class

# %%
explore_image(entry, config.paths.train_images)

# %% [markdown]
# #### Color Mask and Overlay

# %%
explore_color_overlay(entry, config.paths.train_images)

# %% [markdown]
# #### Bounding Box Exploration

# %%
explore_bboxes(entry, config.paths.train_images)

# %% [markdown]
# #### Dataset-wide Report

# %%
dataset_report(entries, config.paths.train_images)
