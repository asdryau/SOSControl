import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
from pathlib import Path

from utils.utils import *
import clip

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

########
#   other data preprocessing function
#######
def load_and_freeze_clip(clip_version="ViT-B/32", device='cpu'):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text_word(clip_model, raw_text, device='cpu'):
    # raw_text - list (batch_size length) of strings with input text prompts
    clip_model.to(device)
    texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
    # return self.clip_model.encode_text(texts).float()

    # copy from encode_text in clip (we use all words)
    x = clip_model.token_embedding(texts).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x

def encode_text_sentence(clip_model, raw_text, device='cpu'):
    # raw_text - list (batch_size length) of strings with input text prompts
    clip_model.to(device)
    texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
    return clip_model.encode_text(texts).float()


if __name__ == "__main__":

    with open("./data/hml3d_text_data.pkl", "rb") as file: 
        text_dict = pickle.load(file)

    clip_model = load_and_freeze_clip(device='cuda')

    # process posetraj format
    text_sentence_all = {}
    # text_word_all = {}
    for k in text_dict:
        text_k = text_dict[k]
        text_sentence = encode_text_sentence(clip_model, text_k, device='cuda').detach().cpu()
        text_word = encode_text_word(clip_model, text_k, device='cuda').detach().cpu()
        # text_sentence_all.append(text_sentence.squeeze(0))  #(512,)
        # text_word_all.append(text_word.squeeze(0)[:20])     #(20,512)
        text_sentence_all[k] = text_sentence
        # text_word_all[k] = text_word[:,:20]
        # print(text_sentence.shape, text_word.shape)
        
    with open(f"./processed_data/text_sentence_all.pkl", 'wb') as handle:
        pickle.dump(text_sentence_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(f"./processed_data/text_word_all.pkl", 'wb') as handle:
    #     pickle.dump(text_word_all, handle, protocol=pickle.HIGHEST_PROTOCOL)