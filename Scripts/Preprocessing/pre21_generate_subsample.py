from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dataset_path, base_path
import RootPath

if __name__ == '__main__':
    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=False)

    # Load dataset
    df = read_dataset(dataset_path, columns=None)

    df_sample = df.sample(frac=0.1, random_state=1234)
    df_sample = df_sample.repartition(partition_size="200MB")

    save_dataset(base_path, df_sample, "SampledDataset")