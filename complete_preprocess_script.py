import os, sys, inspect
import time
import RootPath
import json


def run_script(name, generate_dict, is_test):
    print(f"\nRunning script {name}")
    start_time = time.time()

    command = "python3 Scripts/Preprocessing/" + name
    if not generate_dict:
        command += " --no_dict_generation"
    if is_test:
        command += " --is_test"
    exit_code = os.system(command)
    end_time = time.time()
    print(f"Script {name} ended!\nTime needed: {end_time - start_time}s\n")
    return exit_code


def do_preprocessing(config, all_scripts, generate_dict, is_test) -> None:
    """
    Main function responsible for the preprocessing of the data
    """
    # add root directory for looking for modules
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # guarantee the existence of all needed directories
    for k, directory in config.items():
        if 'path' in k:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    # dump the config as a json to make it available to all scripts
    with open(os.path.join(current_dir, "Scripts", "Preprocessing", "paths.json"), "w+") as write_file:
        json.dump(config, write_file, indent=4, separators=(',', ': '))
    # export envvars usefult to coordinate the scripts
    os.environ['PYTHONPATH'] = current_dir  # project's root dir from where all imports should start
    os.environ['DASK_TEMPORARY_DIRECTORY'] = config['dask_tmp_path']  # where to locate dask's swap

    start = time.time()
    for s in all_scripts:
        exit_code = run_script(s, generate_dict, is_test)
        if exit_code != 0:
            print(f"Exit Code is {exit_code}")
            break
    end = time.time()
    print(f"\n\nTime elapsed for whole script: {end-start}")


if __name__ == '__main__':

    print('Python %s on %s' % (sys.version, sys.platform))

    if RootPath.is_aws():
        print("Detected running on AWS!")
    else:
        print("Running on local")
    print(f"Dataset folder used: {RootPath.get_dataset_path()}")

    # define base path where data can be found
    # all results will be nested in this folder
    data_path = os.path.join(RootPath.get_dataset_path())
    base_path = os.path.join(data_path, 'Preprocessed')
    generate_dict = True
    is_test = False
    all_scripts = [
        "pre00_dataset_to_parquet.py",
        "pre01_map_user_id_features.py",
        "pre02_map_media_features.py",
        "pre03_map_link_id.py",
        "pre04_map_domains_id.py",
        "pre05_map_hashtags_id.py",
        "pre06_map_languages_id.py",
        "pre07_map_tweet_id.py",
        "pre08_map_tweet_type.py",
        "pre09_timestamps.py",
        "pre10_text_preprocessing.py",
        "pre20_merge_all_mapped_features.py",
        # ### "pre21_generate_subsample.py", # should not be used anymore
        "pre22_split_train_val.py"
    ]

    if generate_dict:
        dict_path = os.path.join(base_path, 'Dictionary')
    else:
        dict_path = os.path.join(RootPath.get_dataset_path(), 'Preprocessed', 'Dictionary')

    # define all config paths needed by the subscripts
    config = {
        'original_dataset': os.path.join(data_path, 'part-*'),
        'base_path': base_path,
        'temp_path': os.path.join(base_path, 'Temp'),
        'dict_path': dict_path,
        'train_val_ratio': [0.80, 0.20],
        'dask_tmp_path': os.path.join(base_path, 'Temp', 'dask_tmp'),
    }

    do_preprocessing(config, all_scripts, generate_dict, is_test)
