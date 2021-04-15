from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch_xla.core.xla_model as xm

def train_fn(model, train_dataloader, validation_dataloader, n_epochs, learning_rate, device, best_model_path, seed, freeze_pretrained_encoder=False):
    """
    Train the transformer model for n_epochs and monitor the train and validation losses.
    """

    torch.manual_seed(seed)

    if freeze_pretrained_encoder:
        # Fine-tune from this layer onwards
        # Embedding_Layer has 5 named_parameters, each Transformer layer has 16 named_parameters,
        # and there are total of 24 layers of which we'll freeze the first 12.
        # Hence fine_tune_at = 5 + (12 * 16) = 197
        embed_layers = 1
        embed_parameters = 5
        transformer_layers_to_freeze = 12  # freeze the first n layers
        named_parameters_per_layer = 16
        fine_tune_at = (embed_layers * embed_parameters) + (transformer_layers_to_freeze * named_parameters_per_layer)

        # Get the first 197 of the model's parameter names as a list of tuples
        params_to_freeze = list(model.base_model.named_parameters())[:fine_tune_at]

        # Freeze all the layers before the `fine_tune_at` layer
        for name, param in params_to_freeze:
            print (name, param.data.shape)
            param.requires_grad = False

        model_parameters = filter(lambda p: p.requires_grad, model.base_model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(
            'Total number of trainable parameters in the pretrained encoder after freezing bottom {} transformer layers and {} embed layer: {}'.format(
                transformer_layers_to_freeze, embed_layers, params))
    else:
        print('Model will be fine-tuned by re-training the pretrained transformer')


    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Default optimization
    # Learning rate scheduling is applied after optimizer’s update to adjust the learning rate based on the number of epochs.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * n_epochs)  # number of batches * number of epochs

    train_losses, train_accuracies = [], []  # Store loss and accuracy for plotting
    validation_losses, validation_accuracies = [], []  # Store loss and accuracy for plotting
    best_accuracy = 0.0

    for i in trange(n_epochs, desc="Epoch"):

        # Training mode
        model.train()

        predictions, truth_labels = [], []
        train_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(item.to(device) for item in batch)
            batch_input_ids, batch_input_mask, batch_labels = batch

            optimizer.zero_grad()

            # logits shape: [batch_size, num_classes]
            output = model(batch_input_ids, attention_mask=batch_input_mask, labels=batch_labels)
            loss = output.loss
            logits = output.logits
            #     loss_func = CrossEntropyLoss()
            #     loss = loss_func(logits.view(-1,NUM_LABELS), batch_labels)

            loss.backward()  # back propagate the loss and compute the gradients

            # update the weights
            # for TPU
            xm.optimizer_step(optimizer, barrier=True)  # update the weights
            # scheduler.step()

            train_loss += loss.item()

            batch_logits = logits.detach().cpu().numpy()  # shape: [batch_size, num_classes]
            batch_labels = batch_labels.to('cpu').tolist()  # shape: [batch_size]

            batch_predictions = np.argmax(batch_logits, axis=1).tolist()  # shape: [batch_size]

            predictions.extend(batch_predictions)
            truth_labels.extend(batch_labels)

            if step != 0 and step % 38 == 0:
                print ('Training Batch {} of {}: current_avg_loss={}'.format(step, len(train_dataloader),
                                                                             round(train_loss / step + 1, 2)))
                # print ('Training Batch {} of {}: current_avg_loss={}, lr={}'.format(step, len(train_dataloader), round(train_loss/step+1, 2), round(optimizer.param_groups[0]['lr'], 2)))

        avg_train_acc = round(accuracy_score(y_true=truth_labels, y_pred=predictions), 2) * 100
        avg_train_loss = round(float(train_loss / len(train_dataloader)), 2)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        print("\nAverage Training Loss: {}".format(avg_train_loss))
        print("Average Train Accuracy: {}".format(avg_train_acc))

        # Validation
        predictions, truth_labels = [], []
        val_loss = 0

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        for step, batch in enumerate(validation_dataloader):
            batch = tuple(item.to(device) for item in batch)
            batch_input_ids, batch_input_mask, batch_labels = batch
            with torch.no_grad():
                output = model(batch_input_ids, attention_mask=batch_input_mask, labels=batch_labels)
                loss = output.loss
                logits = output.logits
                val_loss += loss.item()
                #       loss_func = CrossEntropyLoss()
                #       loss = loss_func(logits.view(-1,NUM_LABELS), batch_labels)

                batch_logits = logits.detach().cpu().numpy()  # shape: [batch_size, num_classes]
                batch_labels = batch_labels.to('cpu').tolist()  # shape: [batch_size]

                batch_predictions = np.argmax(batch_logits, axis=1).tolist()  # shape: [batch_size]

                predictions.extend(batch_predictions)
                truth_labels.extend(batch_labels)

        avg_val_acc = round(accuracy_score(y_true=truth_labels, y_pred=predictions), 2) * 100
        avg_val_loss = round(float(val_loss / len(validation_dataloader)), 2)

        if avg_val_acc > best_accuracy:
            print ("Found best model at epoch {}".format(i))
            best_accuracy = avg_val_acc
            torch.save(model, best_model_path)
        #     torch.save(model.state_dict(), 'best-model-parameters.pt')

        validation_losses.append(avg_val_loss)
        validation_accuracies.append(avg_val_acc)

        print("\nAverage Validation Loss: {}".format(avg_val_loss))
        print("Average Validation Accuracy: {}".format(avg_val_acc))

        print("#################################")

    return (train_losses, train_accuracies, validation_losses, validation_accuracies)