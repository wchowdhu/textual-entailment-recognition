import argparse
import os
import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import (XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification)
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch_xla.core.xla_model as xm

from model import Classifier
import data
from train import train_fn
from predict import predict_fn


def plot(epochs, train_losses, train_accuracies, validation_losses, validation_accuracies):
    # plot training and validation losses over all epochs
    n_epochs = epochs + 1
    epochs = range(1, n_epochs)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, validation_losses, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot validation accuracies over all epochs
    epochs = range(1, n_epochs)
    plt.plot(epochs, train_accuracies, 'r', label='Train Accuracy')
    plt.plot(epochs, validation_accuracies, 'm', label='Validation Accuracy')
    plt.title('Average Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')
    parser.add_argument('--bt-file', type=str, default='back_translated.csv') #csv file containing the back translations of input training data
    parser.add_argument('--load-mnli', default=False, action='store_true')
    parser.add_argument('--load-xnli', default=False, action='store_true')
    parser.add_argument('--back-translate', default=False, action='store_true')

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default: 3)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--torch-manual-seed', type=int, default=5, help='torch manual seed (default: 5)')
    parser.add_argument('--patience', type=int, default=0, help='patience (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')  # set the learning rate for the optimizer

    # Model Parameters
    parser.add_argument('--model-name', type=str, default='xlm-roberta-large')
    parser.add_argument('--max-sequence-length', type=int, default=100, help='maximum input sequence length (default: 100)')
    parser.add_argument('--num-labels', type=int, default=3, help='number of output classes (default: 3)')
    parser.add_argument('--freeze-pretrained-encoder', default=False, action='store_true')
    parser.add_argument('--best-model-path', type=str, default='best-model.pt')

    args = parser.parse_args()

    # for TPU
    os.environ["WANDB_API_KEY"] = "0"  # to silence warning
    device = xm.xla_device()
    print('Found TPU at: {}'.format(device))

    # For reproducibility
    np.random.seed(args.seed)

    # Open train and test csv files using pandas library
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    # Split training dataset into two parts - the data we will train the model with and a validation set.
    train_df, validation_df = data.split_data(train_df)

    # Check the number of rows and columns in the subsets after split
    print("Train data shape after split: {} \n".format(train_df.shape))
    print("Validation data shape after split: {} \n".format(validation_df.shape))

    # Augment training data
    train_df = data.augment_data(train_df, test_df, use_xnli=args.load_xnli, use_mnli=args.load_mnli, use_bt=args.back_translate, bt_filepath=args.bt_file)

    # Define the tokenizer to preprocess the input data
    tokenizer = data.define_tokenizer(args.model_name)

    # Batch encode input training data
    train_input = data.encode(train_df, tokenizer, max_len=args.max_sequence_length)
    input_word_ids = train_input['input_word_ids']
    input_mask = train_input['input_mask']
    labels = train_input['labels']
    print("Training input shape: input_word_ids=>{}, input_mask=>{}, labels=>{}".format(input_word_ids.shape, input_mask.shape, labels.shape))

    # Batch encode input validation data
    validation_input = data.encode(validation_df, tokenizer, max_len=args.max_sequence_length)
    validation_word_ids = validation_input['input_word_ids']
    validation_mask = validation_input['input_mask']
    validation_labels = validation_input['labels']
    print("Validation input shape: input_word_ids=>{}, input_mask=>{}, labels=>{}".format(validation_word_ids.shape, validation_mask.shape, validation_labels.shape))

    # Load the input data
    train_dataloader, validation_dataloader = data.get_data_loader(train_input, validation_input, args.batch_size)


    # Build the model by passing in the input params
    model_class = XLMRobertaForSequenceClassification
    model = Classifier(model_class, args.model_name, num_labels=args.num_labels, output_attentions=False, output_hidden_states=False)
    # Send the model to the device
    model.to(device)

    # Train the model
    train_losses, train_accuracies, validation_losses, validation_accuracies = train_fn(model, train_dataloader, validation_dataloader, args.epochs, args.lr, device, args.best_model_path, args.torch_manual_seed, freeze_pretrained_encoder=args.freeze_pretrained_encoder)
    print('Training complete')

    # Plot training and validation losses and accuracies for n_epochs
    # plot(args.epochs, train_losses, train_accuracies, validation_losses, validation_accuracies)

    # Get model predictions on test-set data
    test_input = data.encode(test_df, tokenizer, max_len=args.max_sequence_length, testing=True)
    test_data = TensorDataset(test_input['input_word_ids'], test_input['input_mask'])
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    predictions = predict_fn(test_dataloader, device)

    # Save the test-set predictions
    submission = test_df.id.copy().to_frame()
    submission['prediction'] = predictions
    submission.to_csv("test_predictions.csv", index=False)


