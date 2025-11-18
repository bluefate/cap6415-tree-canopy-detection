# %%
import glob
import os
import shutil


# root (up a directory)
root = os.path.dirname(os.getcwd())

# --- Backup folders ---
backup_root = os.path.join(root, ".codebackup", "root")
backup_root_src = os.path.join(backup_root, "src")

os.makedirs(backup_root, exist_ok = True)
os.makedirs(backup_root_src, exist_ok = True)

# %%
#  --- Copy py files ---
folders_to_copy = [root, os.path.join(root, "notebooks")]  # add more if needed

for folder in folders_to_copy:
    if os.path.isdir(folder):
        # Create a mirror backup folder under .codebackup/root
        relative_name = os.path.relpath(folder, root).replace(os.sep, "_")
        backup_target = os.path.join(backup_root, relative_name)
        os.makedirs(backup_target, exist_ok = True)

        py_files = glob.glob(os.path.join(folder, "*.py"))
        for fname in py_files:
            shutil.copy(fname, backup_target)
            print(f"Copied {fname} -> {backup_target}")


# %%
#  --- merge py files into single file ---

folders_to_process = ["src"]

for folder in folders_to_process:
    source_folder = os.path.join(root, folder)
    backup_root_folder = os.path.join(backup_root, folder)
    os.makedirs(backup_root_folder, exist_ok = True)

    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)

        if os.path.isdir(subfolder_path):
            py_files = glob.glob(os.path.join(subfolder_path, "*.py"))

            if py_files:
                output_file = os.path.join(backup_root_folder, f"_{subfolder}.py")

                with open(output_file, "w") as outfile:
                    for fname in py_files:
                        with open(fname, "r") as infile:
                            outfile.write(f"# From {fname}\n")
                            outfile.write(infile.read())
                            outfile.write("\n\n")

                print(f"Created {output_file}")
