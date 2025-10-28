# %% [markdown]
# ### Version

# %%
# !jupyter --version

# %% [markdown]
# ### Requirements

# %%
import subprocess

# Run pip install and capture output
result = subprocess.run(
    ["pip", "install", "-r", "../requirements.txt"], capture_output=True, text=True
)

# Clean up the output
shortened = result.stdout
shortened = shortened.replace(
    "c:\\users\\johnh\\appdata\\local\\programs\\python\\python312\\lib\\", ""
)
shortened = shortened.replace("from -r ../", "from ")
shortened = shortened.replace("->-r ../", " in ")
shortened = shortened.replace(".txt ", " ")
shortened = shortened.replace("(line", "(ln")
shortened = shortened.replace(")) (", " v")
shortened = shortened.replace("Requirement already satisfied", "Done already")

# Print the cleaned output
print(shortened)

# %%
