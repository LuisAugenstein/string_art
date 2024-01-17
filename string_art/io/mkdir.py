import os


def mkdir(path: str) -> None:
    """
    Creates a directory at the given path if it doesn't already exist.
    """
    paths = path.split('/')
    for i in range(2, len(paths)+1):
        sub_path = '/'.join(paths[:i])
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
