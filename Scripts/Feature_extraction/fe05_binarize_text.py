from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from feature_extraction_utilities import preproc_dict_path, dict_path, temp_output_path, dataset_path

import dask
import dask.dataframe as dd
import numpy as np

out_cols = ['text_nsfw_bool', 'text_kpop_bool', 'text_covid_bool', 'text_sports_bool',
        'text_japanesetrending_bool', 'text_anime_bool', 'text_vtuber_bool', 'text_news_bool',
        'text_myanmar_bool', 'text_genshin_bool', 'text_nintendo_bool', 'text_crypto_bool',
        'text_trending_bool', 'text_love_bool', 'text_slang_bool', 'text_games_bool']
out_frame_name = 'text_bool_features'


if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    # Load dataset
    columns = [
        'text_nsfw_count', 'text_kpop_count', 'text_covid_count', 'text_sports_count',
        'text_japanesetrending_count', 'text_anime_count', 'text_vtuber_count', 'text_news_count',
        'text_myanmar_count', 'text_genshin_count', 'text_nintendo_count', 'text_crypto_count',
        'text_trending_count', 'text_love_count', 'text_slang_count', 'text_games_count'
    ]
    df = read_dataset(dataset_path, columns)

    # Convert the columns
    out = dd.concat(
        [
            df[c].astype(np.bool_).to_frame(c.replace("count", "bool"))
            for c in columns
        ],
        axis=1, ignore_unknown_divisions=True
    )

    # Write the output dataset
    save_dataset(temp_output_path, out, out_frame_name)