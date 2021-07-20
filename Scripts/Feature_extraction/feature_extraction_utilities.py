import RootPath
import os
import json

with open(os.path.join(RootPath.get_root(), "Scripts", "Feature_extraction", "paths.json"), "r") as read_file:
    config = json.load(read_file)

dataset_path = config['data_path']
base_path = config['base_path']
temp_path = config['temp_path']
output_path = os.path.join(base_path, 'All_feature_dataset')
temp_output_path = os.path.join(temp_path, 'Columns')
preproc_dict_path = config['preproc_dict_path']
dict_path = config['dict_path']