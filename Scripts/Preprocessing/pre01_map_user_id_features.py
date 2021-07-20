from preprocessing_utilities import create_and_save_uniques, load_uniques_create_dict_map
from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np

dict_name = "user_id"
out_cols = ["mapped_creator_id", "mapped_engager_id"]
out_frame_name = "mapped_users"

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Map creator_id, engager_id

    # Load dataset
    columns = [
        "raw_feature_creator_id",
        "raw_feature_engager_id"
    ]
    df = read_dataset(dataset_path, columns)

    # Create Dict
    if generate_dict:
        create_and_save_uniques(dict_path,
                                dict_name,
                                df["raw_feature_creator_id"].append(df["raw_feature_engager_id"]))

    # Map the columns
    out = load_uniques_create_dict_map(
        c, dict_path, dict_name,
        df["raw_feature_creator_id"], out_cols[0], np.uint32
    ).to_frame()

    out[out_cols[1]] = load_uniques_create_dict_map(
        c, dict_path, dict_name,
        df["raw_feature_engager_id"], out_cols[1], np.uint32
    )

    # Write the output dataset
    save_dataset(temp_output_path, out, out_frame_name)

