from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torchcrf import CRF


class NLUModule(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_intent_labels = config.num_intent_labels
        self.num_slot_labels = config.num_slot_labels
        self.use_crf = config.use_crf

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot_labels)
        self.crf = CRF(num_tags=config.num_slot_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        slot_labels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        seq_encoding = outputs[0]

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(seq_encoding)
        if self.use_crf and slot_labels is not None:
            crf_loss = self.crf(slot_logits, slot_labels, mask=attention_mask.byte(), reduction='mean')
            crf_loss = -1 * crf_loss  # negative log-likelihood
            return intent_logits, slot_logits, crf_loss
        else:
            return intent_logits, slot_logits, None
