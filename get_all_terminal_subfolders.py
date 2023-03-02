import os

def all_subdirs_search(top_folder):
    subfolders = [f.path for f in os.scandir(top_folder) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(all_subdirs_search(dirname))
    return subfolders

def get_all_terminal_subfolders(top_folder):
    dirs = all_subdirs_search(top_folder)
    terminal_subfolders = dirs.copy()
    for dir_i in dirs:
        for dir_j in dirs:
            if dir_i == dir_j:
                continue
            if dir_i in dir_j:
                terminal_subfolders.remove(dir_i)
                break
    return terminal_subfolders
