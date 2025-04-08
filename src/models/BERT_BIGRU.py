import torch
import torch.nn as nn
from transformers import BertModel

class BERT_BiGRU(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=100, 
                 num_classes=2, dropout_prob=0.2, num_layers=2):
        """
        BERT-BiGRU model for text classification
        
        Args:
            bert_model_name: Name of pretrained BERT model
            hidden_dim: Dimension of BiGRU hidden states
            num_classes: Number of output classes
            dropout_prob: Dropout probability
            num_layers: Number of BiGRU layers (paper found 2 works best)
        """
        super(BERT_BiGRU, self).__init__()
        
        # BERT model for text embeddings
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters if desired (not mentioned in paper)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        bert_hidden_dim = self.bert.config.hidden_size
        
        # Stacked BiGRU layers
        self.bigru = nn.GRU(
            input_size=bert_hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token ids from BERT tokenizer
            attention_mask: Attention mask from BERT tokenizer
            
        Returns:
            logits: Unnormalized model outputs
        """
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the last hidden state (sequence output)
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pass through BiGRU
        gru_output, _ = self.bigru(sequence_output)  # (batch_size, seq_len, hidden_dim*2)
        
        # Get the last hidden state (could also try pooling)
        last_hidden = gru_output[:, -1, :]  # (batch_size, hidden_dim*2)
        
        # Apply dropout
        dropped = self.dropout(last_hidden)
        
        # Classifier
        logits = self.classifier(dropped)  # (batch_size, num_classes)
        
        return logits