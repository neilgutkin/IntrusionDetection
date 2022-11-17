import os, logging
from pprint import pformat
import utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Params:
    pass

# Script params
params = Params()

# Common params
params.hdf_key = 'my_key'
# ---- Small datasets
# params.output_dir = './Datasets/small_datasets/kdd99'
# params.output_dir = './Datasets/small_datasets/nsl_kdd_five_classes'
# params.output_dir = './Datasets/small_datasets/nsl_kdd_five_classes_hard_test_set'
# params.output_dir = './Datasets/small_datasets/ids2017'
# params.output_dir = './Datasets/small_datasets/ids2018'
# ---- Full datasets
# params.output_dir = './Datasets/full_datasets/ids2018'
params.output_dir = './Datasets/small_datasets'

# IDS 2018 params
params.ids2018_datasets_dir = './Datasets/full_datasets'
params.ids2018_files_list = [
                'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
                'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
                #'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
                #'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',
                #'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv'
                #'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
                #'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
                'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
                'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
                'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv'
                ]
params.ids2018_all_X_filename = 'ids2018_all_X.h5'
params.ids2018_all_y_filename = 'ids2018_all_y.h5'
params.ids2018_load_scaler_obj = True
params.ids2018_scaler_obj_path = params.output_dir + '/' + 'partial_scaler_obj.pickle'

params.ids2018_shrink_to_rate = 0.1


def initial_setup(output_dir, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created directory: {}'.format(output_dir))

    # Setup logging
    log_filename = output_dir + '/' + 'run_log.log'

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, 'w+'),
                  logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info('Initialized logging. log_filename = {}'.format(log_filename))

    logging.info('Running script with following parameters\n{}'.format(pformat(params.__dict__)))


def print_dataset_sizes(datasets):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
    logging.info("No. of features = {}".format(X_train.shape[1]))
    logging.info("Training examples = {}".format(X_train.shape[0]))
    logging.info("Validation examples = {}".format(X_val.shape[0]))
    logging.info("Test examples = {}".format(X_test.shape[0]))


def add_additional_items_to_dict(dict, extra_char):
    new_dict = {}
    for key, val in dict.items():
        new_key = key + extra_char
        new_dict[new_key] = val

    dict.update(new_dict)


def prepare_ids2018_datasets_stage_1(params):
    all_ys = {}

    for idx, filename in enumerate(params.ids2018_files_list):
        logging.info('Processing file # {}, filename = {}'.format(idx + 1, filename))

        filepath = params.ids2018_datasets_dir + '/' + filename
        data_df = utility.load_datasets([filepath], header_row=0)
        # data_df = utility.load_datasets([filename], header_row=0, columns_to_read=['Dst Port', 'Flow Duration'])

        logging.info('Sorting by Timestamp')
        data_df.sort_values(by=['Timestamp'], inplace=True)

        cols_to_remove = ['Protocol', 'Timestamp', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP']
        available_cols_to_remove = [col for col in cols_to_remove if col in data_df.columns]
        logging.info('Removing unnecessary columns: {}'.format(available_cols_to_remove))
        data_df.drop(available_cols_to_remove, axis=1, inplace=True)

        logging.info('Removing non-float rows')
        prev_rows = data_df.shape[0]
        utility.remove_non_float_rows(data_df, cols=['Dst Port'])
        logging.info('Removed no. of rows = {}'.format(prev_rows - data_df.shape[0]))

        skip_cols = ['Label']
        logging.info('Converting columns of type object to float')
        utility.convert_obj_cols_to_float(data_df, skip_cols)

        # Remove some invalid values/ rows in the dataset
        # nan_counts = data_df.isna().sum()
        # logging.info(nan_counts)
        logging.info('Removing invalid values (inf, nan)')
        prev_rows = data_df.shape[0]
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_df.dropna(inplace=True)  # Some rows (1358) have NaN values in the Flow Bytes/s column. Get rid of them
        logging.info('Removed no. of rows = {}'.format(prev_rows - data_df.shape[0]))

        X = data_df.loc[:, data_df.columns != 'Label']  # All columns except the last
        y = data_df['Label']

        all_ys[filename] = y

        X_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_X_filename
        y_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_y_filename

        if idx == 0:
            mode = 'w'
        else:
            mode = 'a'  # append

        utility.write_to_hdf(X, X_filename, key=filename, compression_level=5, mode=mode)
        utility.write_to_hdf(y, y_filename, key=filename, compression_level=5, mode=mode)

        logging.info('\n--------------- Processing file complete ---------------\n')

    all_ys_df = pd.concat(list(all_ys.values()))

    # Check class labels
    label_counts, label_perc = utility.count_labels(all_ys_df)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))


def extract_scale_and_write(X_df, y_df, indexes_to_extract, scaler_ob, suffix_str, suffic_index):
    X_extracted = X_df.loc[indexes_to_extract, X_df.columns != 'Label']
    y_extracted = y_df.loc[indexes_to_extract]

    columns = list(range(0, X_extracted.shape[1]))
    X_scaled = utility.scale_dataset(X_extracted, scaler=scaler_ob, columns=columns)

    X_filename = params.output_dir + '/' + 'X_' + suffix_str + '_' + str(suffic_index) + '.h5'
    y_filename = params.output_dir + '/' + 'y_' + suffix_str + '_' + str(suffic_index) + '.h5'

    utility.write_to_hdf(X_scaled, X_filename, params.hdf_key, 5, format='table')
    utility.write_to_hdf(y_extracted, y_filename, params.hdf_key, 5, format='table')


def prepare_ids2018_datasets_stage_2(params):
    X_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_X_filename
    y_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_y_filename

    X_store = pd.HDFStore(X_filename, 'r')
    y_store = pd.HDFStore(y_filename, 'r')

    # print(y_store.keys())

    # Load all y dfs in the y_store, and create corresponding X_info dfs

    logging.info("Loading y dfs in the y_store, and creating corresponding X_info dfs")

    y_all_list = []
    X_info_all_list = []

    for key in y_store.keys():
        print('key = {}'.format(key))
        y_file = y_store[key]
        y_all_list.append(y_file)

        X_info = pd.DataFrame({'file_key': [key] * y_file.shape[0], 'index_in_file': y_file.index})
        X_info_all_list.append(X_info)

    y_all = pd.concat(y_all_list)
    X_info_all = pd.concat(X_info_all_list)

    jump = math.ceil(1 / params.ids2018_shrink_to_rate)
    assert X_info_all.shape[0] == y_all.shape[0]
    X_info_shrunk = X_info_all.iloc[::jump, :]
    y_shrunk = y_all.iloc[::jump]
    logging.info('No. of rows after shrinking dataset: {}'.format(X_info_shrunk.shape[0]))


    # Check class labels
    label_counts, label_perc = utility.count_labels(y_all)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    # ------------

    # Split into 3 sets (train, validation, test)
    logging.info('Splitting dataset set into 3 (train, validation, test)')
    splits = utility.split_dataset(X_info_shrunk, y_shrunk, [0.6, 0.2, 0.2])
    (X_info_train, y_train), (X_info_val, y_val), (X_info_test, y_test) = splits

    # if params.ids2018_load_scaler_obj:
    if False:
        logging.info('Loading partial scaler. path = {}'.format(params.ids2018_scaler_obj_path))
        scaler_obj = utility.load_obj_from_disk(params.ids2018_scaler_obj_path)
    else:
        # Extract training examples at each key and build partial scaler
        logging.info('Extracting training examples at each key and building partial scaler')

        scaler_obj = None
        for i, key in enumerate(X_store.keys()):
            print('key = {}'.format(key))
            X_in_file = X_store[key]

            train_indexes_in_file = X_info_train.loc[X_info_train['file_key'] == key, 'index_in_file']
            X_extracted = X_in_file.loc[train_indexes_in_file, X_in_file.columns != 'Label']

            columns = list(range(0, X_extracted.shape[1]))
            scaler_obj = utility.partial_scaler(X_extracted, scale_type='standard', columns=columns, scaler_obj=scaler_obj)

        utility.save_obj_to_disk(scaler_obj, params.ids2018_scaler_obj_path)
        logging.info("Partial scaler parameters below. \n{}".format(scaler_obj.get_params()))

    for i, key in enumerate(X_store.keys()):
        print('key = {}'.format(key))
        X_in_file = X_store[key]
        y_in_file = y_store[key]

        train_indexes_in_file = X_info_train.loc[X_info_train['file_key'] == key, 'index_in_file']
        val_indexes_in_file = X_info_val.loc[X_info_val['file_key'] == key, 'index_in_file']
        test_indexes_in_file = X_info_test.loc[X_info_test['file_key'] == key, 'index_in_file']

        extract_scale_and_write(X_in_file, y_in_file, train_indexes_in_file, scaler_obj, 'train', i + 1)
        extract_scale_and_write(X_in_file, y_in_file, val_indexes_in_file, scaler_obj, 'val', i + 1)
        extract_scale_and_write(X_in_file, y_in_file, test_indexes_in_file, scaler_obj, 'test', i + 1)


def prepare_ids2018_shrink_dataset(params):
    X_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_X_filename
    y_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_y_filename

    X_store = pd.HDFStore(X_filename, 'r')
    y_store = pd.HDFStore(y_filename, 'r')

    # Load all y dfs in the y_store, and create corresponding X_info dfs
    logging.info("Loading y dfs in the y_store, and creating corresponding X_info dfs")
    y_all_list = []
    X_info_all_list = []

    for key in y_store.keys():
        print('key = {}'.format(key))
        y_file = y_store[key]
        y_all_list.append(y_file)

        X_info = pd.DataFrame({'file_key': [key] * y_file.shape[0], 'index_in_file': y_file.index})
        X_info_all_list.append(X_info)

    y_all = pd.concat(y_all_list)
    X_info_all = pd.concat(X_info_all_list)

    # Shrink datasets
    logging.info('Shrinking dataset')
    X_shrunk_df, y_shrunk_df = shrink_dataset(X_info_all, y_all, params.ids2018_shrink_to_rate, X_store, y_store) #

    # ------------

    # Split into 3 sets (train, validation, test)
    logging.info('Splitting dataset set into 3 (train, validation, test)')
    splits = utility.split_dataset(X_shrunk_df, y_shrunk_df, [0.6, 0.2, 0.2])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    # Scaling
    logging.info('Scaling features (assuming all are numeric)')
    columns = list(range(0, X_train.shape[1]))  # These are the numeric fields to be scaled
    X_train_scaled, scaler = utility.scale_training_set(X_train, scale_type='standard', columns=columns)
    X_val_scaled = utility.scale_dataset(X_val, scaler=scaler, columns=columns)
    X_test_scaled = utility.scale_dataset(X_test, scaler=scaler, columns=columns)

    # Save data files in HDF format
    logging.info('Saving prepared datasets (train, val, test) to: {}'.format(params.output_dir))

    utility.write_to_hdf(X_train_scaled, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_val_scaled, params.output_dir + '/' + 'X_val.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_val, params.output_dir + '/' + 'y_val.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test_scaled, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(splits)


def shrink_dataset(X_info_df, y_df, shrink_to_rate, X_store, y_store):

    jump = math.ceil(1 / shrink_to_rate)
    assert X_info_df.shape[0] == y_df.shape[0]
    X_info_shrunk = X_info_df.iloc[::jump, :]
    y_shrunk = y_df.iloc[::jump]
    logging.info('No. of rows after shrinking dataset: {}'.format(X_info_shrunk.shape[0]))

    # Extract the final records into two dfs X, y
    X_shrunk_dfs = []
    y_shrunk_dfs = []

    for key in X_store.keys():
        print('key = {}'.format(key))
        X_in_file = X_store[key]
        y_in_file = y_store[key]

        indexes_in_file = X_info_shrunk.loc[X_info_shrunk['file_key'] == key, 'index_in_file']

        X_extracted = X_in_file.loc[indexes_in_file, X_in_file.columns != 'Label']
        y_extracted = y_in_file.loc[indexes_in_file]

        X_shrunk_dfs.append(X_extracted)
        y_shrunk_dfs.append(y_extracted)

    X_shrunk_df = pd.concat(X_shrunk_dfs)
    y_shrunk_df = pd.concat(y_shrunk_dfs)

    # Check class labels
    label_counts, label_perc = utility.count_labels(y_shrunk_df)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    return X_shrunk_df, y_shrunk_df





def main():
    initial_setup(params.output_dir, params)
    # --------------------------------------
    # prepare_kdd99_small_datasets(params)
    # prepare_kdd99_full_datasets(params)
    # prepare_nsl_kdd_datasets(params)
    # prepare_ids2017_datasets(params)  # Small subset vs. full is controlled by config flag

    # Following 3 are for preparing the IDS 2018 dataset (20% subset)
    prepare_ids2018_datasets_stage_1(params)
    prepare_ids2018_datasets_stage_2(params)
    prepare_ids2018_shrink_dataset(params)

    logging.info('Data preparation complete')


if __name__ == "__main__":
    main()

