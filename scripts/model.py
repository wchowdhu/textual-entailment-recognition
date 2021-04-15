from torch import nn

class Classifier(nn.Module):
    """
    Define the transformer model to perform multi-class classification
    """
    def __init__(self, model_class, model_name, num_labels=3, output_attentions=False, output_hidden_states=False):

        super(Classifier, self).__init__()

        # Load model; the pretrained model will include a single linear classification layer on top for classification.
        self.model = model_class.from_pretrained(model_name, num_labels=num_labels, output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states)

    
    # Define the forward pass of the network
    def forward(self, input_ids, attention_mask, labels=None, testing=False):
        """
        Perform a forward pass of our model on input features
        """
        if testing: #for test data
            self.output = self.model(input_ids, attention_mask=attention_mask)
        else: #for training and validation data
            self.output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return self.output