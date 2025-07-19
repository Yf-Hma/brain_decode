import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from tokenizers import Tokenizer

os.environ["TOKENIZERS_PARALLELISM"]="false"

current=os.path.dirname(os.path.realpath(__file__))
parent=os.path.dirname(current)
main=os.path.dirname(parent)
sys.path.append(main)

from src.transformers_src.Transformer import DeconvBipartiteTransformerConv


class bold_projector_simple(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.01, device="cuda"):
        super(bold_projector_simple, self).__init__()
        self.linear=nn.Linear(input_size, 256, device=device)
        self.dropout=nn.Dropout(dropout_rate).to(device)
        self.linear_out=nn.Linear(256, output_size, device=device)

    def forward(self, x):
        x=self.linear(x)
        x=self.dropout(x)
        x=self.linear_out(x)
        return x


class bold_projector(nn.Module):
    def __init__(self, encoder, input_size, output_size, dropout_rate=0.05, device="cuda"):
        super(bold_projector, self).__init__()
        self.encoder=encoder

        for param in self.encoder.parameters():
            param.requires_grad=False
        
        self.linear=nn.Linear(input_size, output_size, device=device)
        self.dropout=nn.Dropout(dropout_rate).to(device)
        self.linear_out=nn.Linear(output_size, output_size, device=device)


    def forward(self, x):
        x, _=self.encoder(x)
        x=x[-1]
        x=self.linear(x)
        x=self.dropout(x)
        x=self.linear_out(x)
        return x



class BrainDEC_V0(nn.Module):
    def __init__(
        self,
        configs,
        eeg_encoder_checkpoint,
        device,
        use_lora=False,
        load_in_4bit=False,
        inference_mode=False
    ):
        super().__init__()

        ############# Init Paremeters #############
        self.use_nucleus_sampling=configs.use_nucleus_sampling
        self.num_beams=configs.num_beams
        self.max_new_tokens=configs.max_new_tokens
        self.min_length=configs.min_length
        self.repetition_penalty=configs.repetition_penalty
        self.length_penalty=configs.length_penalty
        self.num_captions=configs.num_captions
        self.temperature=configs.temperature
        self.fixed_instruction=configs.fixed_instruction
        self.max_txt_len=configs.max_txt_len
        self.max_output_txt_len=configs.max_output_txt_len
        self.device=device

        src_eeg_features=configs.src_eeg_features
        time_steps=configs.time_steps
        tokenizer=Tokenizer.from_file("./tools/tokenizer-zuco.json")
        vocab_len=tokenizer.get_vocab_size()
        max_size=configs.max_size
        d_model=configs.d_model
        heads=configs.heads
        d_ff=configs.d_ff
        N=configs.N

        llm_hidden_dim=configs.llm_hidden_dim
        model_name_or_path=configs.LLM_DIR

        ############# Init Brain Signal Encoder #############
        model=DeconvBipartiteTransformerConv(time_steps, src_eeg_features, max_size,\
                                               vocab_len, d_model, d_ff, N, heads, self.device)\
                                               .to(self.device)
        
        checkpoint_filename = eeg_encoder_checkpoint

        if os.path.exists(checkpoint_filename):
            model.load_state_dict(torch.load(checkpoint_filename, weights_only=True))
        else:
            "Warning, Encoder trained checkpoint does not exists."

        ############# Encoder Projector #############
        self.frmi_encoder=bold_projector(model.encoder, d_model, llm_hidden_dim, device=self.device)


        ############# LLM and Tokenizer #############
        self.llm_tokenizer=AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if load_in_4bit:
            bnb_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)

            self.llm_model=AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  quantization_config=bnb_config,
                                                                  local_files_only=True)
        else:
            self.llm_model=AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16,
                                                                  local_files_only=True)
            
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        ############# Init Lora Config #############
        if use_lora:
            #LorA configs
            lora_config=LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                target_modules=["k_proj", "v_proj"],
                lora_dropout=0.05,
                inference_mode=inference_mode,
            )

            self.llm_model=get_peft_model(self.llm_model, lora_config)

        else:
            for param in self.llm_model.parameters():
                param.requires_grad=False
            self.llm_model.eval()
        
        if inference_mode:
            for param in self.llm_model.parameters():
                param.requires_grad=False
            self.llm_model.eval()


    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast=self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len=[]
        llm_tokens={"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones=input_atts[i].sum()
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
        llm_tokens['input_ids']=torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask']=torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len


    def forward (self, sample):
        src_brain_signal, output_text=sample["signal"], sample["text_output"]#.flatten().tolist()
        src_brain_signal=src_brain_signal.type(torch.cuda.FloatTensor).to(self.device)

        input_text=[self.fixed_instruction] *  src_brain_signal.shape[0]

        # BOLD embeddings
        inputs_llm_bold=self.frmi_encoder (src_brain_signal)#.to(device))

        atts_llm_bold=torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        self.llm_tokenizer.padding_side="right"
        self.llm_tokenizer.truncation_side='left'
        text_input_tokens=self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        self.llm_tokenizer.truncation_side='right'
        text_output_tokens=self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in output_text],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len=self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets=llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l]=-100

        # do not apply loss to the image and bold
        empty_targets_bold=(torch.ones(atts_llm_bold.size(), dtype=torch.long).to(self.device).fill_(-100))
        targets=torch.cat([empty_targets_bold, targets], dim=1)

        # Input embeddings
        inputs_embeds=self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds=torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
        attention_mask=torch.cat([atts_llm_bold, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs=self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss=outputs.loss
        return loss

    @torch.no_grad()
    def generate(self,src_brain_signal):

        self.llm_tokenizer.padding_side="left"

        #src_brain_signal=samples
        src_brain_signal=src_brain_signal.type(torch.cuda.FloatTensor).to(self.device)

        # Input text
        prompt=[self.fixed_instruction] *  src_brain_signal.shape[0]

        # Bold embedding
        inputs_llm_bold=self.frmi_encoder (src_brain_signal)#.to(device))

        atts_llm_bold=torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        llm_tokens=self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds=self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds=torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask=torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs=self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.temperature,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                min_length=self.min_length,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                num_return_sequences=self.num_captions,
            )

        output_text=self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text=[text.strip() for text in output_text]

        return output_text

