import os

# Check if a folder is empty.
# Used for checking if VDB already exists.
def check_empty(folder_path):
    if os.path.exists(folder_path):
        return False
    return not os.listdir(folder_path)

def check_exist(folder_path):
    return os.path.exists(folder_path)