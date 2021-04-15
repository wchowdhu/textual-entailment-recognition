# import libraries
import torch
import numpy as np

def predict_fn(test_dataloader, device):
    # load the best training model
    print("Loading the best trained model")
    model = torch.load('best-model.pt')

    model.eval()

    predictions = []

    print('Predicting class labels for the input data...')
    for step, batch in enumerate(test_dataloader):
        batch = tuple(item.to(device) for item in batch)
        batch_input_ids, batch_input_mask = batch
        with torch.no_grad():
            output = model(batch_input_ids, attention_mask=batch_input_mask, testing=True)
            logits = output.logits  # output[0] #[batch_size, num_classes]
            batch_logits = logits.detach().cpu().numpy()  # shape: [batch_size, num_classes]
            batch_predictions = np.argmax(batch_logits, axis=1).tolist()  # shape: [batch_size]
            predictions.extend(batch_predictions)
    print('Complete')
    return predictions