from sklearn.model_selection import train_test_split
from datasets import load_dataset, list_datasets
from tqdm import tqdm
import pandas as pd
from googletrans import Translator
import time
from os import path
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def split_data(train_df):
    # Stratify ensures that each sub-set contains approximately the same percentage of samples of each target class as the original set.
    train_df, validation_df = train_test_split(train_df, stratify=train_df.label.values,
                                               random_state=42,
                                               test_size=0.2, shuffle=True)

    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    return (train_df, validation_df)

def process_xnli_data(test_df, all_keys=False):
    if all_keys:
        split = 'train+validation+test'
    else:
        split = 'validation+test'

    print("Loading XNLI data...")

    dataset = load_dataset('xnli', 'all_languages', split=split)  # returns a Dataset object
    print(dataset)

    entries = []
    for entry in tqdm(dataset):
        hypothesis_langs = entry['hypothesis']['language']  # list of 15 lang string values
        hypothesis_values = entry['hypothesis']['translation']  # list of 15 hypothesis string values

        premise_langs = list(entry['premise'].keys())  # list of 15 lang string values
        premise_values = list(entry['premise'].values())  # list of 15 premise string values

        labels = [entry['label']] * len(hypothesis_langs)  # all 15 languages for the same example have same label

        if premise_langs == hypothesis_langs:  # the languages in premise and hypothesis are in same order
            #             values = list(zip(premise_values, hypothesis_values, hypothesis_langs, labels))
            values = list(zip(premise_values, hypothesis_values, labels))
            entries += values

    #     xnli_df = pd.DataFrame(entries, columns=['premise', 'hypothesis', 'lang_abv', 'label']) #create dataframe for each key
    xnli_df = pd.DataFrame(entries, columns=['premise', 'hypothesis', 'label'])  # create dataframe for each key

    # Get the number of missing data points per column
    missing_values_xnli = xnli_df.isnull().sum()

    print("Number of missing data points per column in XNLI corpus:")
    print (missing_values_xnli)

    # Drop the missing value rows
    xnli_df.dropna(axis=0, inplace=True)
    print("Total number of data examples in XNLI corpus after dropping NA values: {}".format(xnli_df.shape[0]))

    ################## Delete duplicate rows between test_df and xnli_df in xnli corpus #########################
    xnli_df = xnli_df.drop_duplicates()  # drop duplicate rows
    print('Total number of data examples in XNLI corpus after dropping duplicate values: {}'.format(xnli_df.shape[0]))

    print('Test data shape: {}'.format(test_df.shape))
    test_df = test_df.drop_duplicates()  # drop duplicate rows
    print('Test data shape after dropping duplicate rows: {}'.format(test_df.shape))

    df_merge = pd.merge(xnli_df, test_df, on=['premise', 'hypothesis'], how='inner')
    xnli_df = xnli_df.append(df_merge, ignore_index=True)

    xnli_df['duplicated'] = xnli_df.duplicated(subset=['premise', 'hypothesis'],
                                               keep=False)  # keep=False marks the duplicated row with a True
    xnli_df = xnli_df[~xnli_df['duplicated']]  # selects only rows which are not duplicated.

    xnli_df.drop(['duplicated', 'id', 'lang_abv', 'language'], axis=1, inplace=True)  # remove following columns
    ################## Delete duplicate rows between test_df and xnli_df in xnli corpus #########################

    print("XNLI corpus shape: {}".format(xnli_df.shape))

    del dataset  # free up space

    return xnli_df

def process_mnli_data(validation=False):
    if validation:
        keys = ['train', 'validation_matched', 'validation_mismatched']
    else:
        keys = ['train']

    dataset = load_dataset('glue', name='mnli')  # returns a DatasetDict object
    print(dataset)

    df_dict = {}

    for key in tqdm(keys):
        print("Processing MNLI {} data...".format(key))

        hypothesis = dataset[key]['hypothesis']
        premise = dataset[key]['premise']
        labels = dataset[key]['label']

        entries = list(zip(premise, hypothesis, labels))

        df = pd.DataFrame(entries, columns=['premise', 'hypothesis', 'label'])  # create dataframe for each key
        df_dict[key] = df

    if validation:
        mnli_df = pd.concat([df_dict['train'], df_dict['validation_matched'], df_dict['validation_mismatched']],
                            ignore_index=True)
    else:
        mnli_df = df_dict['train']

    # Get the number of missing data points per column
    missing_values_mnli = mnli_df.isnull().sum()

    print("Number of missing data points per column in MNLI corpus:")
    print (missing_values_mnli)

    # Drop the missing value rows from training set
    mnli_df.dropna(axis=0, inplace=True)
    print("Total number of data examples in MNLI corpus after dropping NA values: {}\n".format(mnli_df.shape[0]))

    del dataset  # free up space

    return mnli_df

def back_translate(train_df, target_lang='fr', sample=True, num_samples_per_lang=1000):
    if sample:  # sample input training data to back translate
        train_df = train_df.groupby('language', group_keys=False).apply(
            lambda x: x.sample(min(len(x), num_samples_per_lang))).reset_index(drop=True)

    df_list = []
    limit_before_timeout = 100
    timeout = 5

    translator = Translator()

    # Add functions to back translate input sentences
    def target_translate(x, target_lang):
        translation = translator.translate(x, dest=target_lang)
        return translation.text

    def source_translate(x, source_lang):
        translation = translator.translate(x, dest=source_lang)
        return translation.text

    for i in tqdm(range(len(train_df))):
        entry = train_df.loc[[i]]
        source_lang = entry.lang_abv.values.tolist()[0]
        if source_lang == 'zh':
            # print(googletrans.LANGUAGES)
            source_lang = 'zh-cn'  # 'zh' not in googletrans.LANGUAGES
        if (i != 0) and (i % limit_before_timeout == 0):  # apply timeout after every 100 iterations
            print('Iteration {} of {}'.format(i, len(train_df)))
            time.sleep(timeout)
            # Back translate premise sentence
        entry['premise'] = entry['premise'].apply(lambda x: target_translate(x, target_lang))
        #         time.sleep(0.2)
        entry['premise'] = entry['premise'].apply(lambda x: source_translate(x, source_lang))
        #         time.sleep(0.2)
        # Back translate hypothesis sentence
        entry['hypothesis'] = entry['hypothesis'].apply(lambda x: target_translate(x, target_lang))
        #         time.sleep(0.2)
        entry['hypothesis'] = entry['hypothesis'].apply(lambda x: source_translate(x, source_lang))
        #         time.sleep(0.2)
        df_list.append(entry)

    train_bt = pd.concat(df_list, ignore_index=True)
    print("Shape of back-translated training data: {}".format(train_bt.shape))
    return train_bt

def augment_data(df, test_df, use_xnli=True, use_mnli=True, use_bt=True, bt_filepath=''):
    df_list = []

    if use_bt:
        if path.exists(bt_filepath):
            bt_df = pd.read_csv(bt_filepath)
            print("Shape of back-translated training data: {}".format(bt_df.shape))
        else:
            bt_df = back_translate(df)
        bt_df = bt_df.drop(['id', 'lang_abv', 'language'], axis=1)
        df_list.append(bt_df)
        del bt_df  # free up space
    if use_xnli:
        xnli_df = process_xnli_data(test_df, all_keys=False)
        df_list.append(xnli_df)
        del xnli_df  # free up space
    if use_mnli:
        mnli_df = process_mnli_data(validation=True)
        df_list.append(mnli_df)
        del mnli_df  # free up space

    train_df = df.drop(['id', 'lang_abv', 'language'], axis=1)

    if len(df_list) > 0:  # augment data
        augmented_df = pd.concat(df_list, ignore_index=True)
        train_df = train_df.append(augmented_df, ignore_index=True)

    train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle data
    print("Augmented data shape before removing duplicate rows: {}".format(train_df.shape))
    train_df.drop_duplicates(inplace=True)  # drop duplicate rows
    print("Augmented data shape after removing duplicate rows: {}".format(train_df.shape))
    return train_df

def define_tokenizer(model_name):
    # Download vocabulary from huggingface.co and cache.
    # tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # fast tokenizer
    return tokenizer

def encode(df, tokenizer, max_len=50, testing=False):
    pairs = df[['premise', 'hypothesis']].values.tolist()  # shape=[num_examples]

    print ("Encoding...")
    encoded_dict = tokenizer.batch_encode_plus(pairs, max_length=max_len, padding=True, truncation=True,
                                               add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
    print ("Complete")

    if not testing:
        target_labels = torch.tensor(df.label.values)
    else:
        target_labels = None

    inputs = {
        'input_word_ids': encoded_dict['input_ids'],
        'input_mask': encoded_dict['attention_mask'],
        'labels': target_labels}

    return inputs

def get_data_loader(train_input, validation_input, batch_size):
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_input['input_word_ids'], train_input['input_mask'], train_input['labels'])
    train_sampler = RandomSampler(train_data)  # Samples elements randomly. If without replacement, then sample from a shuffled dataset. If with replacement, then user can specify num_samples to draw.
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_input['input_word_ids'], validation_input['input_mask'], validation_input['labels'])
    validation_sampler = SequentialSampler(validation_data)  # Samples elements sequentially, always in the same order.
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return (train_dataloader, validation_dataloader)
