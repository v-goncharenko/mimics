from sys import path


# patching path is dirty trick but acceptable for notebooks
dest_path = '..'

if dest_path not in path:
    path.insert(0, dest_path)
