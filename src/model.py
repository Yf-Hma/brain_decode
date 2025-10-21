from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig

from CLIP import clip

########## Aligenemnt Block 1 ##############
class alignment_block(nn.Module):
    def __init__(self, encoder, input_size, output_size, src_features_max, freeze_encoder, dropout_rate=0.01, device="cuda"):
        super(alignment_block, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout_rate).to(device)
        self.linear_out = nn.Linear(output_size, output_size, device=device)
        self.device = device

        self.src_features_max = src_features_max

        # Freeze the encoder
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, x):
        if x.shape[2] != self.src_features_max:
            padded = torch.zeros(x.shape[0], x.shape[1], self.src_features_max - x.shape[2]).to(self.device)
            x = torch.cat([x,padded], dim = 2)

        x, _ = self.encoder (x)
        x = x[-1]
        x = self.linear(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        
        return x
    

########## Aligenemnt Block 2 ##############
class alignment_block_aug(nn.Module):
    def __init__(self, encoder, input_size, output_size, src_features_max, freeze_encoder, dropout_rate=0.01, device="cuda"):
        super(alignment_block_aug, self).__init__()
        
        self.linear_input = nn.Linear(src_features_max, input_size, device=device)
        self.linear = nn.Linear(input_size, output_size, device=device)
        self.dropout = nn.Dropout(dropout_rate).to(device)
        self.linear_out = nn.Linear(output_size, output_size, device=device)
        self.device = device

        self.src_features_max = src_features_max
        self.layer_norm = nn.LayerNorm(input_size)

        # Adding learnable parameters to the encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(input_size, nhead=4, batch_first=True)
        self.encoder_raw = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)

        #Freeze the encoder
        self.encoder = encoder
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, x):

        y = self.linear_input(x)
        y = self.encoder_raw(y)
        y = self.layer_norm(y)

        x, _ = self.encoder (x)
        x = x[-1]
        x = self.layer_norm(x)

        x = x[:, :3, :] + y
        #x = torch.cat((x[:, :y.shape[1], :], y), 2)
        x = self.layer_norm(x)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.linear_out(x)
        
        return x
    
########## BrainDEC models ##############
class BrainDEC_V0(nn.Module):
    def __init__(
        self,
        encoder_model,
        configs,
        src_features_max,
        freeze_encoder = True,
        max_txt_len=128,
        max_output_txt_len=256,
        lora = False,
        inference_mode = False,
        load_in_4bit = False,
        device = "cuda",
        align = "normal"
    ):
        super().__init__()

        self.configs = configs
        d_model = configs.d_model
        self.device = device

        model_name_or_path = configs.LLM_PATH

        # Alignment_block for fMRI Encoder adaptation
        llm_hidden_dim = configs.llm_hidden_dim     
        
        if align == "normal":   
            self.frmi_encoder = alignment_block(encoder_model, d_model, llm_hidden_dim,  src_features_max, freeze_encoder, device=self.device).to(self.device)
        elif align == "aug":   
            self.frmi_encoder = alignment_block_aug(encoder_model, d_model, llm_hidden_dim,  src_features_max, freeze_encoder, device=self.device).to(self.device)
            
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        if lora:
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

        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

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

        # Input text
        input_text = [self.configs.fixed_instruction] *  len (sample["text_output"])
        
        # Target text
        output_text = sample["text_output"]
        
        # BOLD embeddings
        inputs_llm_bold = self.frmi_encoder (sample["signal"].to(self.device).float())
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        # Tokenization
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

        # Forward and compute loss
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets)

        loss = outputs.loss
        return loss


    @torch.no_grad()
    def generate(self, samples):

        self.llm_tokenizer.padding_side = "left"
        prompt = [self.configs.fixed_instruction] * len (samples["text_output"])

        # Bold embedding        
        inputs_llm_bold = self.frmi_encoder (samples["signal"].to(self.device).float())    
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
                do_sample=True,
                top_p=self.configs.top_p,
                temperature=self.configs.temperature,
                num_beams=self.configs.num_beams,
                max_new_tokens=self.configs.max_new_tokens,
                min_length=self.configs.min_length,
                repetition_penalty=self.configs.repetition_penalty,
                length_penalty=self.configs.length_penalty,
                num_return_sequences=self.configs.num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text


class BrainDEC_V1(nn.Module):
    def __init__(
        self,
        encoder_model,
        configs,
        src_features_max,
        max_txt_len=128,
        max_output_txt_len=256,
        lora = False,
        inference_mode = False,
        load_in_4bit = False,
        device = "cuda"
    ):
        super().__init__()

        d_model = configs.d_model

        model_name_or_path = configs.LLM_PATH
        self.device = device
        self.configs = configs

        # fMRI encoder
        llm_hidden_dim = configs.llm_hidden_dim    
        self.frmi_encoder = alignment_block(encoder_model, d_model, llm_hidden_dim,  src_features_max, True, device=self.device).to(self.device)

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
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        if lora:
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

        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len


    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast("cuda", dtype=dtype)
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
        input_text = [self.configs.fixed_instruction_with_input_text + "'"  + a  + "' : " for a in sample["text_input"]]

        # BOLD embeddings
        inputs_llm_bold = self.frmi_encoder (sample["signal"].to(self.device))#.to(device))
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
    def generate(self, samples):

        self.llm_tokenizer.padding_side = "left"

        image = samples["image"]
        bs = image.size(0)

        prompt = [self.configs.fixed_instruction_with_input_text + "'"  + a  + "' : " for a in samples["text_input"]]

        # Bold embedding
        inputs_llm_bold = self.frmi_encoder (samples["signal"].to(self.device))#.to(device))
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
                top_p=self.configs.top_p,
                temperature=self.configs.temperature,
                num_beams=self.configs.num_beams,
                max_new_tokens = self.configs.max_new_tokens,
                min_length=self.configs.min_length,
                repetition_penalty=self.configs.repetition_penalty,
                length_penalty=self.configs.length_penalty,
                num_return_sequences=self.configs.num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text


class BrainDEC_V2(nn.Module):

    def __init__(
        self,
        encoder_model,
        configs,
        src_features_max,
        max_txt_len=128,
        max_output_txt_len=256,
        load_in_4bit = False,
        lora = False,
        inference_mode = False,
        device = "cuda"):
        super().__init__()

        model_name_or_path = configs.LLM_PATH

        self.device = device
        self.configs = configs

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False

        self.clip_model.eval()

        d_model = configs.d_model

        model_name_or_path = configs.LLM_PATH

        # fMRI encoder
        llm_hidden_dim = configs.llm_hidden_dim      
        self.frmi_encoder = alignment_block(encoder_model, d_model, llm_hidden_dim,  src_features_max, True, device=self.device).to(self.device)
          

        self.max_output_txt_len = max_output_txt_len

        for param in self.frmi_encoder.parameters():
            param.requires_grad = False
        self.frmi_encoder.eval()


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

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))


        if lora:
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

        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.vision_llm_proj = nn.Linear(512, configs.llm_hidden_dim).to(self.device)

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len


    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
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

        #input_text = ["En se basant sur ce contenu, réponds en Français à la phrase suivante '" + a  + "' : " for a in sample["text_input"]]
        input_text = [self.configs.fixed_instruction_with_input_text + "'"  + a  + "' : " for a in sample["text_input"]]
        
        output_text = sample["text_output"]

        # Images embedding and  alignement
        image_features = self.clip_model.encode_image(sample["image"].to(self.device))
        image_features= torch.unsqueeze(image_features, dim=1).to(torch.float32).to(self.device)
        input_llm_image = self.vision_llm_proj (image_features)
        atts_llm_image = torch.ones(input_llm_image.size()[:-1], dtype=torch.long).to(self.device)

        # BOLD embedding and  alignement
        inputs_llm_bold = self.frmi_encoder (sample["signal"].to(self.device))#.to(device))
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

        # do not apply loss to the query tokens
        empty_targets_bold = (torch.ones(atts_llm_bold.size(), dtype=torch.long).to(self.device).fill_(-100))
        empty_targets_image = (torch.ones(atts_llm_image.size(), dtype=torch.long).to(self.device).fill_(-100))

        targets = torch.cat([empty_targets_image, empty_targets_bold, targets], dim=1)

        # Input embeddings
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])


        inputs_embeds = torch.cat([ input_llm_image, inputs_llm_bold, inputs_embeds], dim=1)
        attention_mask = torch.cat([ atts_llm_image, atts_llm_bold, llm_tokens['attention_mask']], dim=1)

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
    def generate(self, samples):

        self.llm_tokenizer.padding_side = "left"

        image = samples["image"]
        bs = image.size(0)

        prompt = [self.configs.fixed_instruction_with_input_text + "'"  + a  + "' : " for a in samples["text_input"]]
        #prompt = ["En se basant sur ce contenu, réponds en Français à la phrase suivante '" + a  + "' : " for a in samples["text_input"]]

        # Signal encoding
        inputs_llm_bold = self.frmi_encoder (samples["signal"].to(self.device))#.to(device))
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

        # Image embedding alignement
        image_features = self.clip_model.encode_image(samples["image"].to(self.device))
        image_features= torch.unsqueeze(image_features, dim=1)
        input_llm_image = self.vision_llm_proj (image_features.to(torch.float32))
        atts_llm_image = torch.ones(input_llm_image.size()[:-1], dtype=torch.long).to(self.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([input_llm_image, inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_image, atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=self.configs.top_p,
                temperature=self.configs.temperature,
                num_beams=self.configs.num_beams,
                max_new_tokens = self.configs.max_new_tokens,
                min_length=self.configs.min_length,
                repetition_penalty=self.configs.repetition_penalty,
                length_penalty=self.configs.length_penalty,
                num_return_sequences=self.configs.num_captions,
            )

        #outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
