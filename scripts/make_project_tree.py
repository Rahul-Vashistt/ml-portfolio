import os

def print_tree(path, indent="", ignore_ext=(".pyc", ".pyd", ".exe"),
               ignore_dirs=(".git", ".venv", ".vscode", "__pycache__", "dist", "build"),
               ignore_suffix=("egg-info", "dist-info"),
               max_depth=None, level=0):
    """
    Recursively prints a visual tree structure of files and directories.

    Features:
    - Skips unwanted directories (e.g., `.git`, `.venv`, `__pycache__`).
    - Skips files with certain extensions (e.g., `.pyc`, `.pyd`, `.exe`).
    - Skips directories ending with specific suffixes (e.g., `egg-info`, `dist-info`).
    - Supports limiting recursion depth via `max_depth`.
    - Uses tree-like connectors (`├──`, `└──`, `│`) for clear hierarchy.

    Args:
        path (str): The root directory to start printing from.
        indent (str): Indentation string used internally for formatting.
        ignore_ext (tuple): File extensions to skip.
        ignore_dirs (tuple): Directory names to skip.
        ignore_suffix (tuple): Directory suffixes to skip.
        max_depth (int or None): Maximum depth of recursion (None = unlimited).
        level (int): Current recursion depth (used internally).

    Example:
        >>> print_tree("my_project", max_depth=2)

        my_project/
        ├── README.md
        ├── src/
        │   ├── main.py
        │   └── utils.py
        └── tests/
            └── test_main.py
    """
    if max_depth is not None and level > max_depth:
        return

    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(indent + "└── [Permission Denied]")
        return

    for i, item in enumerate(items):
        full_path = os.path.join(path, item)
        connector = "└── " if i == len(items) - 1 else "├── "

        if os.path.isdir(full_path):
            if (item in ignore_dirs or item.endswith(ignore_suffix)):
                continue

            print(indent + connector + item + "/")
            new_indent = indent + ("    " if i == len(items) - 1 else "│   ")
            print_tree(full_path, new_indent, ignore_ext, ignore_dirs,
                       ignore_suffix, max_depth, level + 1)
        else:
            if item.endswith(ignore_ext):
                continue
            print(indent + connector + item)


# Example usage
print_tree(r"D:\Languages\Projects\ml-portfolio", max_depth=3, indent="")
