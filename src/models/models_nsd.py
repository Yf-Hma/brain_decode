import os
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig



current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)

from src.models.brain_clip import BrainCLIP, ImageEncoder
from src.transformers_src.Transformer import DeconvBipartiteTransformerConv, Transformer
import src.configs.nsd.configs_nsd as configs



class bold_projector(nn.Module):
    def __init__(self, encoder, input_size, output_size, dropout_rate=0.01, device="cuda"):
        super(bold_projector, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, output_size, device=device)
        self.linear_out = nn.Linear(output_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout_rate).to(device)

    def forward(self, x):
        x, _ = self.encoder (x)
        x = x[-1]
        batch_size, T, N1 = x.shape
        x = self.linear(x)
        return x



class BrainDEC_V0(nn.Module):
    def __init__(
        self,
        src_fmri_features,
        device,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False,
        inference_mode = False
    ):
        super().__init__()

        #src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        vocab_len = configs.vocab_len
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        self.coco_captions = np.load(configs.coco_annotation_file_path)

        model_name_or_path = configs.LLM_DIR
        self.device = device

        model = Transformer(time_steps, src_fmri_features, max_size,\
                                               vocab_len, d_model, d_ff, N, heads, self.device)\
                                               .to(self.device)


        emb_dim = configs.emb_dim
        d_model = emb_dim

        llm_hidden_dim = configs.llm_hidden_dim

        self.frmi_encoder = bold_projector (model.encoder, d_model, llm_hidden_dim, device=device)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)

            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  quantization_config=bnb_config,
                                                                  local_files_only=True)
        else:
            if configs.LLM_name == "gemma9b":

                self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                    device_map=self.device,
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16,
                                                                    attn_implementation='eager',
                                                                    local_files_only=True)
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                    device_map=self.device,
                                                                    trust_remote_code=True,
                                                                    #torch_dtype=torch.bfloat16,
                                                                    local_files_only=True)


        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # LorA configs
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.01,
            inference_mode = inference_mode,
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

        if inference_mode:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model.eval()


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
        src_bold_signal, trg_ids = sample[0], sample[2].flatten().tolist()
        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)
        output_text = self.coco_captions[trg_ids]
        # Input text
        input_text = [configs.fixed_instruction] * src_bold_signal.shape[0]

        # BOLD embeddings
        #embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        #embeddings = embeddings[-1]

        inputs_llm_bold = self.frmi_encoder (src_bold_signal)#.to(device))

        #inputs_llm_bold = self.llm_proj (embeddings)
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
                labels=targets,)

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

        #src_bold_signal = samples
        src_bold_signal = samples.type(torch.cuda.FloatTensor).to(self.device)

        # Input text
        prompt = [configs.fixed_instruction] *  src_bold_signal.shape[0]

        # Bold embedding
        #bold_embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        #bold_embeddings = bold_embeddings[-1]


        inputs_llm_bold = self.frmi_encoder (src_bold_signal)#.to(device))

        #inputs_llm_bold = self.llm_proj (bold_embeddings)

        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)


        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
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


class MllmBrainToTextVIT(nn.Module):

    def __init__(
        self,
        device,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False
    ):
        super().__init__()

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        vocab_len = configs.vocab_len
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        self.coco_captions = np.load('../tools/COCO_73k_annots.npy')

        model_name_or_path = configs.LLM_DIR
        self.device = device

        n_heads = configs.n_heads
        n_layers = configs.n_layers
        width = configs.width
        max_seq_length = configs.max_size
        emb_dim = configs.emb_dim
        vocab_size = configs.vocab_size
        img_size = configs.img_size
        patch_size = configs.patch_size
        n_channels = configs.n_channels
        vit_width = configs.vit_width
        vit_layers = configs.vit_layers
        vit_heads = configs.vit_heads
        d_model = emb_dim

        #model.load_state_dict(torch.load(encoder_path, weights_only=True))

        self.frmi_encoder = ImageEncoder(vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, emb_dim).to(self.device)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)

            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  quantization_config=bnb_config,
                                                                  local_files_only=True)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16,
                                                                  local_files_only=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # for name, param in self.llm_model.named_parameters():
        #     param.requires_grad = False

        # for name, layer in self.llm_model.model.layers.named_children():
        #     if name in ["1", "2"]:
        #         for param in layer.parameters():
        #             param.requires_grad = True

        # LorA configs
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.llm_proj = nn.Linear(d_model, 4096).to(self.device)


        self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

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

        src_bold_signal, trg_ids = sample[0], sample[2].flatten().tolist()

        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)

        #src_bold_signal = src_bold_signal.unsqueeze (1)
        output_text = self.coco_captions[trg_ids]
        # Input text
        input_text = [configs.fixed_instruction] * src_bold_signal.shape[0]

        # BOLD embeddings
        embeddings = self.frmi_encoder (src_bold_signal)#.to(device))

        #embeddings = embeddings[-1]

        inputs_llm_bold = self.llm_proj (embeddings)

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

        #src_bold_signal = samples
        src_bold_signal = samples.type(torch.cuda.FloatTensor).to(self.device)

        # Input text
        prompt = ["Provide a caption based on this content: "] *  src_bold_signal.shape[0]

        # Bold embedding
        bold_embeddings = self.frmi_encoder (src_bold_signal)#.to(device))

        inputs_llm_bold = self.llm_proj (bold_embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
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

class MllmBrainToText_Deconv(nn.Module):

    def __init__(
        self,
        device,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False
    ):
        super().__init__()

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        vocab_len = configs.vocab_len
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        self.coco_captions = np.load('../tools/COCO_73k_annots.npy')

        model_name_or_path = configs.LLM_DIR
        self.device = device
        encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "BipartiteTransformerConv_%d_%s.pt"%(src_fmri_features, configs.type))
        #encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "brain_clip_deconv_200.pth")

        n_heads = configs.n_heads
        n_layers = configs.n_layers
        width = configs.width
        max_seq_length = configs.max_size
        emb_dim = configs.emb_dim
        vocab_size = configs.vocab_size
        img_size = configs.img_size
        patch_size = configs.patch_size
        n_channels = configs.n_channels
        vit_width = configs.vit_width
        vit_layers = configs.vit_layers
        vit_heads = configs.vit_heads
        d_model = emb_dim


        model = DeconvBipartiteTransformerConv(time_steps, src_fmri_features, max_size,\
                                               vocab_len, d_model, d_ff, N, heads, self.device)\
                                               .to(self.device)

        #model = model.float()
        #model.load_state_dict(torch.load(encoder_path, weights_only=True))

        self.frmi_encoder = model.encoder

        # self.frmi_encoder.train()

        # for param in self.frmi_encoder.parameters():
        #     param.requires_grad = False
        # self.frmi_encoder.eval()


        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)

            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  quantization_config=bnb_config,
                                                                  local_files_only=True)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16,
                                                                  local_files_only=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # for name, param in self.llm_model.named_parameters():
        #     param.requires_grad = False

        # for name, layer in self.llm_model.model.layers.named_children():
        #     if name in ["1", "2"]:
        #         for param in layer.parameters():
        #             param.requires_grad = True


        # LorA configs
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.01,
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.llm_proj = nn.Linear(d_model, 4096).to(self.device)


        self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

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

        src_bold_signal, trg_ids = sample[0], sample[2].flatten().tolist()

        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)

        output_text = self.coco_captions[trg_ids]
        # Input text
        input_text = [configs.fixed_instruction] * src_bold_signal.shape[0]

        # BOLD embeddings
        embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        embeddings = embeddings[-1]
        inputs_llm_bold = self.llm_proj (embeddings)
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

        src_bold_signal = samples
        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)

        # Input text
        prompt = ["Provide a caption based on this content: "] *  src_bold_signal.shape[0]

        # Bold embedding
        bold_embeddings, _ = self.frmi_encoder (src_bold_signal.to(self.device))#.to(device))
        bold_embeddings = bold_embeddings[-1]
        inputs_llm_bold = self.llm_proj (bold_embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)


        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
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

class MllmBrainToTextV0_clip(nn.Module):

    def __init__(
        self,
        device,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False,
        inference_mode=False
    ):
        super().__init__()

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        vocab_len = configs.vocab_len
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        self.coco_captions = np.load('../tools/COCO_73k_annots.npy')

        model_name_or_path = configs.LLM_DIR
        self.device = device

        n_heads = configs.n_heads
        n_layers = configs.n_layers
        width = configs.width
        max_seq_length = configs.max_size
        emb_dim = configs.emb_dim
        vocab_size = configs.vocab_size
        #img_size = configs.img_size

        patch_size = configs.patch_size
        n_channels = configs.n_channels
        vit_width = configs.vit_width
        vit_layers = configs.vit_layers
        vit_heads = configs.vit_heads
        d_model = emb_dim

        img_size = (time_steps, src_fmri_features)

        model = BrainCLIP(emb_dim, vit_width, img_size, patch_size,
                          n_channels, vit_layers, vit_heads, vocab_size,
                          width, max_seq_length, n_heads, n_layers, device, type="normal")

        encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "brain_clip_normal_30.pth")

        checkpoint = torch.load(encoder_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            model.load_state_dict(checkpoint["model"], strict=False)

        self.frmi_encoder = model.image_encoder

        # for param in self.frmi_encoder.parameters():
        #     param.requires_grad = False

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)

            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  quantization_config=bnb_config,
                                                                  local_files_only=True)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16,
                                                                  local_files_only=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # LorA configs
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.01,
            inference_mode = inference_mode,
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

        if inference_mode:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            #self.llm_model.eval()


        if configs.LLM_name == "llama3b":
            self.llm_proj = nn.Linear(d_model, 3072).to(self.device)
        elif configs.LLM_name == "vicuna15":
            self.llm_proj = nn.Linear(d_model, 5120).to(self.device)
        else:
            self.llm_proj = nn.Linear(d_model, 4096).to(self.device)


        #self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

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
        src_bold_signal, trg_ids = sample[0], sample[2].flatten().tolist()
        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)
        output_text = self.coco_captions[trg_ids]
        # Input text
        input_text = [configs.fixed_instruction] * src_bold_signal.shape[0]

        # BOLD embeddings
        embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        embeddings = embeddings[-1]

        inputs_llm_bold = self.llm_proj (embeddings)
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

        src_bold_signal = samples
        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)

        # Input text
        prompt = ["Provide a caption based on this content: "] *  src_bold_signal.shape[0]

        # Bold embedding
        bold_embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        bold_embeddings = bold_embeddings[-1]

        inputs_llm_bold = self.llm_proj (bold_embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)


        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
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

class MllmBrainToText_Deconv_clip(nn.Module):

    def __init__(
        self,
        device,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False
    ):
        super().__init__()

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        vocab_len = configs.vocab_len
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        self.coco_captions = np.load('../tools/COCO_73k_annots.npy')

        model_name_or_path = configs.LLM_DIR
        self.device = device

        model = DeconvBipartiteTransformerConv(time_steps, src_fmri_features, max_size,\
                                               vocab_len, d_model, d_ff, N, heads, self.device)\
                                               .to(self.device)

        encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "brain_clip_deconv_200.pth")

        n_heads = configs.n_heads
        n_layers = configs.n_layers
        width = configs.width
        max_seq_length = configs.max_size
        emb_dim = configs.emb_dim
        vocab_size = configs.vocab_size
        img_size = configs.img_size
        patch_size = configs.patch_size
        n_channels = configs.n_channels
        vit_width = configs.vit_width
        vit_layers = configs.vit_layers
        vit_heads = configs.vit_heads
        d_model = emb_dim

        model = BrainCLIP(emb_dim, vit_width, img_size, patch_size,
                          n_channels, vit_layers, vit_heads, vocab_size,
                          width, max_seq_length, n_heads, n_layers, device, type="normal")

        checkpoint = torch.load(encoder_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            model.load_state_dict(checkpoint["model"], strict=False)

        self.frmi_encoder = model.image_encoder

        # for param in self.frmi_encoder.parameters():
        #     param.requires_grad = False
        # self.frmi_encoder.eval()

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16)

            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  quantization_config=bnb_config,
                                                                  local_files_only=True)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                  device_map=self.device,
                                                                  trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16,
                                                                  local_files_only=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(d_model, 4096).to(self.device)


        #self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

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

        src_bold_signal, trg_ids = sample[0], sample[2].flatten().tolist()

        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)

        output_text = self.coco_captions[trg_ids]
        # Input text
        input_text = [configs.fixed_instruction] * src_bold_signal.shape[0]

        # BOLD embeddings
        embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        embeddings = embeddings[-1]

        inputs_llm_bold = self.llm_proj (embeddings)

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

        src_bold_signal = samples
        src_bold_signal = src_bold_signal.type(torch.cuda.FloatTensor).to(self.device)

        # Input text
        prompt = ["Provide a caption based on this content: "] *  src_bold_signal.shape[0]

        # Bold embedding
        bold_embeddings, masks = self.frmi_encoder (src_bold_signal)#.to(device))
        bold_embeddings = bold_embeddings[-1]

        inputs_llm_bold = self.llm_proj (bold_embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)


        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
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
