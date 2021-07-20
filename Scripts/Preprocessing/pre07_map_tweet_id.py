from preprocessing_utilities import create_and_save_uniques, load_uniques_create_dict_map
from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np
import RootPath

dict_name = "tweet_id"
out_cols = ["mapped_tweet_id"]

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Map Languages

    # Load dataset
    df = read_dataset(dataset_path, ["raw_feature_tweet_id"])

    # Create Dict
    if generate_dict:
        create_and_save_uniques(dict_path,
                                dict_name,
                                df["raw_feature_tweet_id"])

    # Map the columns
    out = load_uniques_create_dict_map(c, dict_path, dict_name,
                                       df["raw_feature_tweet_id"], out_cols[0], np.uint32).to_frame()

    # Write the output dataset
    save_dataset(temp_output_path, out)

