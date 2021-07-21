# ACM RecSys Challenge 2021
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="Recommender System 2018 Challenge Polimi" />
</p>
<p align="center">
    <img width="10%" src="https://seeklogo.com/images/T/twitter-logo-A84FE9258E-seeklogo.com.png" />
</p>
<p align="center">
  <a href="http://recsys.deib.polimi.it">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" />
  </a>
</p>
## About the Challenge

From the [Challenge](https://recsys-twitter.com/) website: 

*The RecSys Challenge 2021 will be organized by [Politecnico di Bari](http://www.en.poliba.it/), [ETH Zürich](https://ethz.ch/en.html), [Jönköping University](https://ju.se/en), and the data set will be provided by [Twitter](https://twitter.com). The challenge focuses on a real-world task of tweet engagement  prediction in a dynamic environment. For 2021, the challenge considers  four different engagement types: Likes, Retweet, Quote, and replies.*

## Team Members

We participated in the challenge as Trial&Error, a team of 8 MSc students from Politecnico di Milano:

* **[Luca Carminati](https://github.com/LCarmi)**

* **[Giacomo Lodigiani](https://github.com/Lodz97)**

* **[Pietro Maldini](https://github.com/pm390)**

* **[Samuele Meta](https://github.com/SamueleMeta)**

* **[Stiven Metaj](https://github.com/StivenMetaj)**

* **[Arcangelo Pisa](https://github.com/pisa97)**

* **[Alessandro Sanvito](https://github.com/Alexdruso)**

* **[Mattia Surricchio](https://github.com/mattiasu96)**

We worked under the supervision of:
* **[Fernando B. Peréz Maurera](https://github.com/fernandobperezm)**
* **[Cesare Bernardis](https://github.com/cesarebernardis)**
* **[Maurizio Ferrari Dacrema](https://github.com/maurizioFD)**

## Paper

Section to be added

## Requirements

In order to run the code it is necessary to have:

- **Python**: version 3.8.
- **CUDA**.
- **Docker**
- **nvidia-docker** correctly initialized. See [link](https://github.com/NVIDIA/nvidia-docker) for instructions.

Install the python dependecies with the following bash command:

```
pip install -r requirements.txt
```
Alternatively, the following command can be usedì:
```
pip3 install dask==2021.4 distributed==2021.4 "dask[complete]" bokeh scipy numpy pyarrow fastparquet transformers lightgbm optuna xgboost catboost
```

### Download the dataset

If you do not have the dataset, you can download it from here: https://recsys-twitter.com/data/show-downloads (registration is required).

After the dataset has been downloaded (without modifying the file  names) , it can be decompressed throught the following commands:
```
lzop --verbose --delete -d *.lzo
rm *.lzo.index
```
These commands will create a series of `part-*` files containing the uncompressed data. These data will have then to be moved to the base data folder used by the code to run. The code expects the validation data parts to be in a subfolder named `official_valid` of the folder containing all the training parts.

See the next Configuration section to have proper instructions on how to configure the data paths for the code.

### Computation Requirements

Computations on a single chunk can be run locally. However to perform computations over the full dataset, we relied on some AWS EC2 instances. In particular:

* `c5.24xlarge` or `r5.24xlarge` instances with Ubuntu 20 AMI have been used for all the phases, expect for Target Encodings computations
* `g4dn.12xlarge` instances with Ubuntu 18 DeepLearning AMI have been employed for the Target Encodings computations

A rather large disk will be needed to store the dataset and the various nested intermediate results. We used a 2TB drive. The final dataset used for training and testing the models has been save on a separate 200GB drive.

## Configuration

Top level scripts have to be configured manually in specific points of the code, in order to run the full pipeline. Specific interventions needed will be described in the *Run the code* section of this README.

Here we provide an overview of the top-level files contatining important configurations.

* ***Rootpath.py*** contains the pathing logic needed to access correctly the data to be used. In particular, we distinguished the data paths to be used when running on AWS machines and locally. When running locally, the full dataset is expected to be in the Dataset subfolder of the root folder of the project. When running on AWS, we used `/home/ubuntu/data` as data folder. The fact of running on AWS is detected through a reading o SSH environmental variables.
* ***complete_preprocess_script.py*** and ***complete_feature_extraction_script.py*** are the top-level scripts used to manage the phases of preprocessing and feature extraction. More specifically, they coordinate the subsequential launch of the various scripts composing our pipeline. The important configuration flags and paths are defined in the `if __name__ == '__main__':` part of the scripts. In particular:
  * **generate_dict** flag specifies whether the dictionaries used to map the features are to be created, saved, and then mapped onto the dataset under considration, or whether those dicts have already been generated and can be found in the `dict_path` folder.
  * **is_test** is a flag specifying whether we are running the scripts on data containing also the target columns.
  * **config** dict contains all the paths needed to run the scripts. Default paths should work without modifications if you stick to our default paths.
  * a **list of scripts** to be run, specifying the order of the scripts that will be run subsequentially

## Run the Code

### Data Preprocessing and Feature Extraction

The pipeline of code to be run to obtain the final model, has been splitted into various parts to make it more resilient and to improve code usage. In the following, we assume that the 2TB drive is monted on DATA_FOLDER, which is the path we use to store the data (eg. the default one if using AWS is `/home/ubuntu/data`), while the MODEL_DATA_FOLDER is the 200GB drive (by default mounted on `/home/ubuntu/new`)

1. **Preprocess all dataset:** in this phase, we preprocess all data, mapping unique ids to integers, creating text features, and splitting the dataset in train and validation.

   To perform this step:

   * Make sure `generate_dict=True` in `complete_preprocess_script.py`

   * Run the script

     ```
     python3 complete_preprocess_script.py
     ```

2. **Generate memory-based features dictionaries:** in this phase, we use the *Train* data to compute the memory based features.

   To perform this step:

   * Make sure `generate_dict=True` in `complete_feature_extraction_script.py`.  The default paths already specified in  `complete_feature_extraction_script.py` 's config dictionary should be correct 

   * Make sure all and only the scripts from `fe01_follower_features.py` to `fe_32a_target_encoding_split_cols.py` are uncommented

   * Run the script

     ```
     python3 complete_feature_extraction_script.py
     ```

   * Generate Target Encoding features:

     * Run the docker container

       ```
       # docker run --gpus all -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --mount type=bind,source=DATA_FOLDER,target=/rapids/notebooks/host     rapidsai/rapidsai:0.19-cuda11.0-runtime-ubuntu18.04-py3.7
       ```

     * access to the jupyter lab at port `8888`, and load the notebook `Scripts/Feature_extraction/TE_Efficient.ipynb`

     * run all the notebook's cells sequentially

3. **Map memory-based features over an unseen dataset:** in this phase, we map the previously computed features over the *Valid* data to create the dataset that will be used for training and testing the model.

   To perform this step:

   * Make sure `generate_dict=False` in `complete_feature_extraction_script.py`.  The default paths already specified in  `complete_feature_extraction_script.py` 's config dictionary should be correct 

   * Make sure all the scripts from `fe01_follower_features.py` to `fe_33_target_encoding_mapping.py` are uncommented

   * Run the script

     ```
     python3 complete_feature_extraction_script.py
     ```

   At this point, the preprocessed and mapped data reay for training and testing will be available at the path: `DATA_FOLDER/Preprocessed/Valid/FeatureExtraction/All_feature_dataset/Valid_with_TE`

4. **Create the final splits:** in this phase we split the data in folders and prepare to use them in the model training and validation.

   To perform this step:

   * be sure to have mounted the 200GB drive on MODEL_DATA_FOLDER.
   * If any  paths are defferent from the default ones, modify `final_splits.ipynb` accordingly.
   * execute `final_splits.ipynb` notebook to repartition the dataset in the possible splits available.

### Model Training and Testing

In the following, we describe the passeges to be made to train and test our models. We assume the 200GB drive to be mounted on DATA_FOLDER =  `/home/ubuntu/new`.
Run the following script to tune the lightGBM model:

```
python3 train_lightgbm_CPU_parametric_tuning.py
```
The above script contains a variable ```idx``` which is used to select which model to tune:
```
idx = 0 -> tune like model
idx = 1 -> tune retweet model
idx = 2 -> tune reply model
idx = 3 -> tune comment model
```
The script outputs a ```.txt``` file with the parameters of the best model for the selected class.

Once the models have been tuned, the best parameters must copied inside ```train_lightgbm_CPU_parametric_class.py```. The script contains 4 dictionaries: ```params_like, params_retweet, params_reply, params_comment```, one for each lightGBM model. 
Change the ```idx``` value inside ```train_lightgbm_CPU_parametric_class.py``` to select which model to train, then run:
```
python3 train_lightgbm_CPU_parametric_class.py
```

The script will output two .txt files, one with the performances of the trained model, the other with the saved model.
