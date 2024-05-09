import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration as _T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Union, Tuple
from model.table_model import TableEncoder
from model.maxpooling import MaxPooling

class T5ForConditionalGeneration(_T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        # if use_prompt > 0, use RPM, if use_super > 0, use RSM
        self.use_super = config.use_super
        self.use_prompt = config.use_prompt
        self.table_loss_alpha = config.table_loss_alpha
        
        # use context embedding and ResNet-style CNN to encode the word-level Embedding to Table Embedding
        if self.use_prompt > 0 or self.use_super > 0:
            config.table_encoder = 'resnet'
            config.method = 't5-prompt'
            self.table = TableEncoder(config)
            self.pooling_func = MaxPooling()
            
            # Considering different task has different types of sentiments and aspect category
            # The classifier_number of RSM should be different
            # In triplet, the types are: A, O, S_neg, S-neu, S-pos, E-neg, E-neu, S-pos
            # In quad, the types are: A, O, S_neg, S-neu, S-pos, E-neg, E-neu, S-pos, S_cate_0, E_cate_0, ..., S_cate_n, E_cate_n
            # The number is 2 + 2 * sentiment types + 2 * aspect category types (which should be 0 in ASTE)
            if config.task == 'triplet':
                self.classifier_number = 8
            elif config.task == 'quad':
                if 'rest' in config.dataset:
                    self.classifier_number = 30
                else:
                    self.classifier_number = 72
            
            self.table_classifier = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, self.classifier_number))
        self.config = config
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        table_labels: Optional[torch.LongTensor] = None,
        table_labels_S: Optional[torch.LongTensor] = None,
        table_labels_E: Optional[torch.LongTensor] = None,
        table_labels_new: Optional[torch.LongTensor] = None,
        pairs_true = None,
        ID = None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        
        if self.use_prompt >= 1 or self.use_super >= 1:
            if not self.training:
                if past_key_values == None:
                    table_embeds, table_mask = self.table(hidden_states, attention_mask)
                    past_key_values = ((table_embeds, table_mask), )
                else:
                    table_embeds, table_mask = past_key_values[0]
            else:
                table_embeds, table_mask = self.table(hidden_states, attention_mask)

            if self.training:
                output = self.table_classifier(table_embeds)
                if self.table_loss_alpha > 0:
                    table_loss_func = nn.BCEWithLogitsLoss(weight=table_mask, reduction='sum')
                    table_loss = 0
                    for i in range(output.shape[-1]):
                        table_loss += table_loss_func(output[:,:,:,i],table_labels_new[:,:,:,i].float())/table_mask.sum()
                else:
                    table_loss_func = nn.BCEWithLogitsLoss(weight=table_mask)
                    table_loss = 0
                    for i in range(output.shape[-1]):
                        table_loss += table_loss_func(output[:,:,:,i],table_labels_new[:,:,:,i].float())
            
            if self.use_prompt >= 1:
                table_x = self.pooling_func(table_embeds, attention_mask, dim=1)
                table_y = self.pooling_func(table_embeds, attention_mask, dim=2)

                prompt_embeds = torch.max(table_x, table_y)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if self.use_prompt >= 1:
            decoder_inputs_embeds, decoder_attention_mask = self.add_prompt(decoder_input_ids, prompt_embeds, attention_mask)
        else:
            decoder_inputs_embeds = self.shared(decoder_input_ids)
            decoder_attention_mask = None

        # Decode
        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if self.use_prompt >= 1:
            self.remove_prompt(decoder_outputs, prompt_embeds)

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None

        if (labels is not None) and (self.training):
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            if self.use_super >= 1:
                loss = loss + table_loss*abs(self.table_loss_alpha)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            past_key_values=past_key_values
        )

    def add_prompt(self, decoder_input_ids, prompt_embeds, attention_mask):
        batch_size, mask_seq_length = decoder_input_ids.size()

        inputs_embeds = self.shared(decoder_input_ids)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        decoder_attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        decoder_attention_mask = torch.cat([attention_mask, decoder_attention_mask], dim=1)

        return inputs_embeds, decoder_attention_mask

    def remove_prompt(self, decoder_outputs, prompt_embeds):

        prompt_length = prompt_embeds.size(1)

        decoder_outputs.last_hidden_state = decoder_outputs.last_hidden_state[:, prompt_length:]

        if decoder_outputs.hidden_states is not None:
            for every_hidden_states in decoder_outputs.hidden_states:
                every_hidden_states = every_hidden_states[:, prompt_length:]