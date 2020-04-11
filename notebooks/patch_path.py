from sys import path


dest_path = '..'

# dirty trick but acceptable for notebooks
if dest_path not in path:
    path.insert(0, dest_path)
