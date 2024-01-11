import os

root_path = os.path.abspath(__file__).replace('/string_art/io/root_path.py', '')


def get_project_dir(name_of_the_run: str) -> str:
    project_dir = f'{root_path}/data/outputs/{name_of_the_run}'
    if not os.path.exists(project_dir):
        os.mkdir(project_dir)
    return project_dir
