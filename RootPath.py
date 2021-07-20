import pathlib as pl
import os


def get_root():
    return pl.Path(__file__).parent.absolute()


def is_aws():
    #return True
    return True if os.environ.get("SSH_CLIENT") else False


def get_dataset_path():
    if is_aws():
        return "/home/ubuntu/data"
    else:
        return pl.Path(__file__).parent.joinpath("Dataset/").absolute()
