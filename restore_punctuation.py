#!/usr/bin/env python3
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
import tqdm
import torch
import string

punctuation = list(string.punctuation)
punctuation_tokens = []

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", low_cpu_mem_usage=True)
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")["validation"]
audio = dataset[0]["audio"]["array"]
text = dataset[0]["text"].lower()

inputs = processor(audio, return_tensors="pt")
input_features = inputs["input_features"]


def get_punctuation_tokens(tokenizer):
    punctuation_tokens = []
    for i in tqdm.tqdm(range(len(tokenizer))):
        if tokenizer.convert_ids_to_tokens(i) in punctuation:
            punctuation_tokens.append(i)
    return punctuation_tokens


punctuation_tokens = get_punctuation_tokens(processor.tokenizer)

# whisper always starts as follows
decoder_start_tokens = [50257, 50362]

lower_words = [f"Ġ{word}" for word in text.split()]
target_lower_words = [processor.tokenizer(word, add_special_tokens=False).input_ids for word in lower_words]

upper_words = [f"Ġ{word.capitalize()}" for word in text.split()]
target_upper_words = [processor.tokenizer(word, add_special_tokens=False).input_ids for word in upper_words]

all_upper_words = [f"Ġ{word.upper()}" for word in text.split()]
target_all_upper_words = [processor.tokenizer(word, add_special_tokens=False).input_ids for word in all_upper_words]


with torch.no_grad():
    encoder_hidden_states = model.model.encoder(input_features).last_hidden_state

# start tokens for English
decoder_start_ids = torch.tensor(decoder_start_tokens, dtype=torch.long)[None, :]
current_ids = decoder_start_ids.broadcast_to(encoder_hidden_states.shape[:1] + decoder_start_ids.shape[1:])


for l_w, u_w, up_w in zip(target_lower_words, target_upper_words, target_all_upper_words):
    is_finished = False
    while not is_finished:
        logits = model(decoder_input_ids=current_ids, encoder_outputs=encoder_hidden_states).logits[:, -1]
        top_k_10 = logits.topk(10).indices
        import ipdb; ipdb.set_trace()

