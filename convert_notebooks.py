#convert_notebooks.py
"""
Convert notebooks with:
pipenv run python convert_notebooks.py

Ignore hash files and force conversion of all notebooks:
pipenv run python convert_notebooks.py --force
"""

POSTS_DIRECTORY = "_posts/notebooks"
NOTEBOOKS_DIRECTORY = "_notebooks"
ASSETS_DIRECTORY = "assets/notebooks"
REPLACEMENT_STRINGS = [("<IPython.core.display.Javascript object>", "")]


import os
import sys
import getopt
import shutil
import hashlib
from glob import glob
from subprocess import check_output


def convert_notebooks(force_build: bool):
    notebooks = notebook_paths()
    create_assets_path()
    create_posts_path()

    if not force_build:
        notebooks = [x for x in notebooks if notebook_changed(x)]

    for notebook_path in notebooks:
        convert_to_markdown(notebook_path)
        move_to_posts(notebook_path)
        move_assets(notebook_path)
        write_hash_file(notebook_path)

    print(f"*** Converted {len(notebooks)} notebooks ***")


def notebook_changed(notebook_path: str) -> bool:
    path = hash_path(notebook_path)

    if not os.path.exists(path):
        return True

    hash1 = compute_hash(notebook_path)

    with open(path, "rt") as fh:
        hash2 = fh.read()
        return hash1 != hash2


def compute_hash(notebook_path: str) -> str:
    with open(notebook_path, "rb") as fh:
        contents = fh.read()
        return hashlib.sha256(contents).hexdigest();


def hash_path(notebook_path: str) -> str:
    return notebook_path + ".sha256"


def write_hash_file(notebook_path: str):
    hash = compute_hash(notebook_path)
    path = hash_path(notebook_path);

    with open(path, "wt") as fh:
        fh.truncate()
        fh.write(hash)


def notebook_paths():
    paths = []
    for folder, _, _ in os.walk(NOTEBOOKS_DIRECTORY):
        if ".ipynb_checkpoints" not in folder:
            paths.extend(glob(os.path.join(folder, "*.ipynb")))
    return paths


def create_assets_path():
    if not os.path.exists(ASSETS_DIRECTORY):
        os.makedirs(ASSETS_DIRECTORY)


def create_posts_path():
    if not os.path.exists(POSTS_DIRECTORY):
        os.makedirs(POSTS_DIRECTORY)


def convert_to_markdown(notebook_path: str):
    command = f"jupyter nbconvert {notebook_path} --to markdown"
    check_output(command, shell = True)
    postprocess_markdown(notebook_path)


def postprocess_markdown(notebook_path: str):
    md_path = markdown_path(notebook_path)

    with open(md_path, "rt") as fh:
        contents = fh.read()

    contents = replace_strings(contents)
    contents = update_markdown_paths(notebook_path, contents)

    with open(md_path, "wt") as fh:
        fh.truncate()
        fh.write(contents)


def markdown_path(notebook_path: str) -> str:
    return notebook_path[:-6] + ".md"


def move_to_posts(notebook_path: str):
    md_path = markdown_path(notebook_path)
    fname = os.path.basename(md_path)
    posts_path = os.path.join(POSTS_DIRECTORY, fname)
    shutil.move(md_path, posts_path)


def move_assets(notebook_path: str):
    path = assets_path(notebook_path)
    directory = os.path.basename(path)
    dest = os.path.join(ASSETS_DIRECTORY, directory)

    if os.path.exists(dest):
        shutil.rmtree(dest)

    if os.path.exists(path):
        shutil.move(path, dest)


def replace_strings(contents: str) -> str:
    for (x, y) in REPLACEMENT_STRINGS:
        contents = contents.replace(x, y)
    return contents


def assets_path(notebook_path: str) -> str:
    return notebook_path[:-6] + "_files"


def update_markdown_paths(notebook_path: str, contents: str) -> str:
    dest = ["![Jupyter Notebook Plot]("]

    if not ASSETS_DIRECTORY.startswith("/"):
        dest.append("/")

    dest.append(ASSETS_DIRECTORY)

    if not ASSETS_DIRECTORY.endswith("/"):
        dest.append("/")

    return contents.replace("![png](", "".join(dest))


def main():
    options, _ = getopt.getopt(sys.argv[1:], "", ["force"])
    force_build = False

    for (option, _) in options:
        if option == "--force":
            force_build = True

    convert_notebooks(force_build)


if __name__ == "__main__":
    main()
