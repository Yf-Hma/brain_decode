import os
import sys
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from src.transformers_src.Transformer import DeconvBipartiteTransformerConv, Transformer

from peft import get_peft_model, LoraConfig


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)

from src.transformers_src.Transformer import DeconvBipartiteTransformerConv
import src.configs.perceived.configs as configs


class bold_projector(nn.Module):
    def __init__(self, encoder, input_size, output_size, src_fmri_features_max, dropout_rate=0.01, device="cuda"):
        super(bold_projector, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, output_size, device=device)
        self.linear_out = nn.Linear(output_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout_rate).to(device)

        self.src_fmri_features_max = src_fmri_features_max

        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x):
        if x.shape[2] != self.src_fmri_features_max:
            padded = torch.zeros(x.shape[0], x.shape[1], self.src_fmri_features_max - x.shape[2])
            x = torch.cat([x,padded], dim = 2)

        x, _ = self.encoder (x)
        x = x[-1]
        x = self.dropout(x)
        x = self.linear(x)
        return x




class BrainDEC_V0(nn.Module):

    def __init__(
        self,
        fmri_encoder_path,
        src_fmri_features,
        max_txt_len=128,
        lora=False,
        max_output_txt_len=512,
    ):
        super().__init__()

        d_model = configs.d_model
        d_ff = configs.d_ff
        heads = configs.heads
        N = configs.N
        time_steps = configs.time_steps
        max_size = configs.max_size

        tokenizer = Tokenizer.from_file("./tools/tokenizer-perceived.json")
        vocab_len = tokenizer.get_vocab_size()
        model_name_or_path = configs.LLM_DIR
        self.device = "cuda"

        model = DeconvBipartiteTransformerConv(time_steps, configs.src_fmri_features_max, max_size, vocab_len, d_model, d_ff, N, heads, self.device).to(self.device)
        model = model.float()
        model.load_state_dict(torch.load(fmri_encoder_path, weights_only=True))

        llm_hidden_dim = configs.llm_hidden_dim

        self.frmi_encoder = bold_projector(model.encoder, d_model, llm_hidden_dim,  configs.src_fmri_features_max, device=self.device)


        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              device_map=self.device,
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.bfloat16,
                                                              local_files_only=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))


        if lora:
            # LorA configs
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                target_modules=["k_proj", "v_proj"],
                lora_dropout=0.01
            )

            self.llm_model = get_peft_model(self.llm_model, lora_config)


        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.llm_proj = nn.Linear(d_model, configs.llm_hidden_dim).to(self.device)

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len


    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len


    def forward (self, sample):

        output_text = sample["text_output"]
        # Input text
        input_text = ["<end_of_bold>" for a in sample["text_input"]]

        # BOLD embeddings
        inputs_llm_bold = self.frmi_encoder (sample["bold_signal"].to(self.device))#.to(device))

        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in output_text],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )


        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)
        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the image and bold
        empty_targets_bold = (torch.ones(atts_llm_bold.size(), dtype=torch.long).to(self.device).fill_(-100))
        #empty_targets_image = (torch.ones(atts_llm_image.size(), dtype=torch.long).to(self.device).fill_(-100))

        targets = torch.cat([empty_targets_bold, targets], dim=1)

        # Input embeddings
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm_bold, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        return loss


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=configs.use_nucleus_sampling,
        num_beams=configs.num_beams,
        max_new_tokens = configs.max_new_tokens,
        min_length=configs.min_length,
        top_p=configs.top_p,
        repetition_penalty=configs.repetition_penalty,
        length_penalty=configs.length_penalty,
        num_captions=configs.num_captions,
        temperature=configs.temperature):

        self.llm_tokenizer.padding_side = "left"

        prompt = ["<end_of_bold>"] *  len (samples["text_input"])

        # Bold embedding
        inputs_llm_bold = self.frmi_encoder (samples["bold_signal"].to(self.device))#.to(device))

        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens = max_new_tokens,
                #max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
