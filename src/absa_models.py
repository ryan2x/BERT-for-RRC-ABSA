import torch
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertPreTrainingHeads
import logging

logger = logging.getLogger(__name__)


class BertForMTPostTraining(PreTrainedBertModel):
    def __init__(self, config):
        super(BertForMTPostTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, mode, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, start_positions=None, end_positions=None):
        
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        if mode=="review":
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
            if masked_lm_labels is not None and next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))        
                next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                
                return total_loss
            else:
                return prediction_scores, seq_relationship_score

        elif mode=="squad":
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                qa_loss = (start_loss + end_loss) / 2
                return qa_loss
            else:
                return start_logits, end_logits
        else:
            raise Exception("unknown mode.")


class BertForSequenceLabeling(PreTrainedBertModel):
    def __init__(self, config, num_labels=3):
        super(BertForSequenceLabeling, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

