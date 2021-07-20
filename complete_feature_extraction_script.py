import os, sys, inspect
import time
import RootPath
import json


def run_script(name, generate_dict, is_test):
    print(f"\nRunning script {name}")
    start_time = time.time()
    command = "python3 Scripts/Feature_extraction/" + name
    if not generate_dict:
        command += " --no_dict_generation"
    if is_test:
        command += " --is_test"
    exit_code = os.system(command)
    end_time = time.time()
    print(f"Script {name} ended!\nTime needed: {end_time - start_time}s\n")
    return exit_code


def do_feature_extraction(config, all_scripts, generate_dict, is_test) -> None:
    """
    Main function responsible for the feature extraction of the data
    """
    # add root directory for looking for modules
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # guarantee the existence of all needed directories
    for k, directory in config.items():
        if 'path' in k:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    # dump the config as a json to make it available to all scripts
    with open(os.path.join(current_dir, "Scripts", "Feature_extraction", "paths.json"), "w+") as write_file:
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

    preproc_dict_path = os.path.join(RootPath.get_dataset_path(), 'Preprocessed', 'Dictionary')
    generate_dict = True
    is_test = False

    if generate_dict:
        data_path = os.path.join(RootPath.get_dataset_path(), 'Preprocessed', 'Train')
        base_path = os.path.join(data_path, 'FeatureExtraction')
        dict_path = os.path.join(base_path, 'Dictionary')
    else:
        data_path = os.path.join(RootPath.get_dataset_path(), 'Preprocessed', 'Valid')
        base_path = os.path.join(data_path, 'FeatureExtraction')
        dict_path = os.path.join(RootPath.get_dataset_path(), 'Preprocessed', 'Train', 'FeatureExtraction', 'Dictionary')
    all_scripts = [
        'fe01_follower_features.py',
        'fe02_user_hashtags.py',
        'fe03_categorical_combo.py',
        'fe20_merge_all_features.py',
        'fe_32a_target_encoding_split_cols.py',
        # 'fe_33_target_encoding_mapping'
    ]

    # define all config paths needed by the subscripts
    config = {
        'data_path': data_path,
        'base_path': base_path,
        'temp_path': os.path.join(base_path, 'Temp'),
        'preproc_dict_path': preproc_dict_path,
        'dict_path': dict_path,
        'dask_tmp_path': os.path.join(base_path, 'Temp', 'dask_tmp'),
    }

    do_feature_extraction(config, all_scripts, generate_dict, is_test)


