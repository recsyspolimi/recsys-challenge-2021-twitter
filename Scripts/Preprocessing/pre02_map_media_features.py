from preprocessing_utilities import create_and_save_uniques, load_uniques_create_dict_map
from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np
import dask
import dask.dataframe as dd

out_cols = ["number_of_photo", "number_of_gif", "number_of_video", 'presence_of_photo', 'presence_of_gif', 'presence_of_video']
out_frame_name = "mapped_media"


def functional_map_media(media_series: dask.dataframe.Series) -> dask.dataframe.DataFrame:
    # Map the feature
    n_photo = media_series.str.count('Photo')
    n_gif = media_series.str.count('GIF')
    n_video = media_series.str.count('Video')

    out_media = dd.concat(
        [
            n_photo.astype(np.uint8).to_frame(name='number_of_photo'),
            n_gif.astype(np.uint8).to_frame(name='number_of_gif'),
            n_video.astype(np.uint8).to_frame(name='number_of_video'),
            n_photo.astype(np.bool_).to_frame(name='presence_of_photo'),
            n_gif.astype(np.bool_).to_frame(name='presence_of_gif'),
            n_video.astype(np.bool_).to_frame(name='presence_of_video'),
        ],
        axis=1, ignore_unknown_divisions=True
    )

    return out_media


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    ### Map media
    # Load dataset
    df = read_dataset(dataset_path, ["raw_feature_tweet_media"])

    # Do functional mapping
    media_df = functional_map_media(df["raw_feature_tweet_media"])

    # Write the output dataset
    save_dataset(temp_output_path, media_df, out_frame_name)