import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from packaging import version
import transformers
import torch
import torch.nn as nn
from torch.optim import Adam
from PIL import Image
from CLIP import clip
from tokenizers import Tokenizer

from src.transformers_src.Transformer import DeconvBipartiteTransformerConv
import src.configs as configs


class MllmBrainToTextV0(nn.Module):

    def __init__(
        self,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False
    ):
        super().__init__()

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        tokenizer = Tokenizer.from_file("tools/tokenizer-trained.json")
        vocab_len = tokenizer.get_vocab_size()

        model_name_or_path = configs.LLM_DIR
        self.device = "cuda"
        encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "DeconvBipartiteTransformerConv_%d_%s.pt"%(src_fmri_features, configs.type))

        model = DeconvBipartiteTransformerConv(time_steps, src_fmri_features, max_size,\
                                               vocab_len, d_model, d_ff, N, heads, self.device)\
                                               .to(self.device)
        model = model.float()
        model.load_state_dict(torch.load(encoder_path, weights_only=True))
        self.frmi_encoder = model.encoder

        for param in self.frmi_encoder.parameters():
            param.requires_grad = False
        self.frmi_encoder.eval()


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
        output_text = sample["text_output"]
        # Input text
        [configs.fixed_instruction + "'" + a  + "'"   + ": " for a in sample["text_input"]]

        # BOLD embeddings
        embeddings, masks = self.frmi_encoder (sample["bold_signal"].to(self.device))#.to(device))
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

        prompt = [configs.fixed_instruction] *  len (samples["text_input"])

        # Bold embedding
        bold_embeddings, _ = self.frmi_encoder (samples["bold_signal"].to(self.device))#.to(device))
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
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text





class MllmBrainToText(nn.Module):

    def __init__(
        self,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False
    ):
        super().__init__()

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        tokenizer = Tokenizer.from_file("tools/tokenizer-trained.json")
        vocab_len = tokenizer.get_vocab_size()

        model_name_or_path = configs.LLM_DIR
        self.device = "cuda"

        # self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "DeconvBipartiteTransformerConv_%d_%s.pt"%(src_fmri_features, configs.type))
        model = DeconvBipartiteTransformerConv(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, self.device).to(self.device)
        # self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        model = model.float()
        model.load_state_dict(torch.load(encoder_path, weights_only=True))
        self.frmi_encoder = model.encoder

        for param in self.frmi_encoder.parameters():
            param.requires_grad = False
        self.frmi_encoder.eval()

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


        self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

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
        [configs.fixed_instruction + "'" + a   + "'"  + ": " for a in sample["text_input"]]

        # BOLD embeddings
        embeddings, masks = self.frmi_encoder (sample["bold_signal"].to(self.device))#.to(device))
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

        image = samples["image"]
        bs = image.size(0)

        prompt = [configs.fixed_instruction + "'" + a  + "' : " for a in samples["text_input"]]

        # Bold embedding
        bold_embeddings, _ = self.frmi_encoder (samples["bold_signal"].to(self.device))#.to(device))
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




class MllmBrainToTextV2(nn.Module):

    def __init__(
        self,
        max_txt_len=configs.max_txt_len,
        max_output_txt_len=configs.max_output_txt_len,
        load_in_4bit = False):
        super().__init__()

        model_name_or_path = configs.LLM_DIR

        self.device = "cuda"

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        self.clip_model.eval()
        #self.visual_encoder.train = disabled_train

        src_fmri_features = configs.src_fmri_features
        time_steps = configs.time_steps
        max_size = configs.max_size
        d_model = configs.d_model
        heads = configs.heads
        d_ff = configs.d_ff
        N = configs.N

        tokenizer = Tokenizer.from_file("tools/tokenizer-trained.json")
        vocab_len = tokenizer.get_vocab_size()

        self.device = "cuda"

        # self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        encoder_path = os.path.join (configs.MODELS_TRAIN_DIR, "DeconvBipartiteTransformerConv_%d_%s.pt"%(src_fmri_features, configs.type))
        model = DeconvBipartiteTransformerConv(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, self.device).to(self.device)
        # self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        model = model.float()
        model.load_state_dict(torch.load(encoder_path, weights_only=True))
        self.frmi_encoder = model.encoder

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
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(d_model, 4096).to(self.device)


        self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

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

        input_text = [configs.fixed_instruction + "'" + a  + "'"   + ": " for a in sample["text_input"]]
        output_text = sample["text_output"]

        # Images embedding and  alignement
        image_features = self.clip_model.encode_image(sample["image"].to(self.device))
        image_features= torch.unsqueeze(image_features, dim=1).to(torch.float32).to(self.device)
        input_llm_image = self.vision_llm_proj (image_features)
        atts_llm_image = torch.ones(input_llm_image.size()[:-1], dtype=torch.long).to(self.device)

        # BOLD embedding and  alignement
        embeddings, masks = self.frmi_encoder (sample["bold_signal"].to(self.device))#.to(device))
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

        image = samples["image"]
        bs = image.size(0)

        prompt = [configs.fixed_instruction + "'" + a  + "' :" for a in samples["text_input"]]

        bold_embeddings, _ = self.frmi_encoder (samples["bold_signal"].to(self.device))#.to(device))
        bold_embeddings = bold_embeddings[-1]
        # Image embedding alignement
        inputs_llm_bold = self.llm_proj (bold_embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)

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
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([input_llm_image, inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_image, atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens = max_new_tokens,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
