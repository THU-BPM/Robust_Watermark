import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def replace_synonyms(text, target_prob=0.2):
    words = text.split()
    num_words = len(words)
    real_replace = 0

    replaceable_indices = []
    
    # First pass: Identify replaceable words
    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        synonyms = [syn for syn in synonyms if len(syn.lemmas()) > 1]
        if synonyms:
            replaceable_indices.append(i)

    # Calculate the number of words to replace
    num_to_replace = int(min(target_prob, len(replaceable_indices) / num_words) * num_words)

    # Randomly select words to replace
    indices_to_replace = random.sample(replaceable_indices, num_to_replace)

    # Perform replacement
    for i in indices_to_replace:
        synonyms = wordnet.synsets(words[i])
        synonyms = [syn for syn in synonyms if len(syn.lemmas()) > 1]
        if synonyms:
            chosen_syn = random.choice(synonyms)
            words[i] = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
            real_replace += 1

            
    print(f"Target Replace Rate: {target_prob}, Real Replace Rate: {real_replace / num_words}")
    
    return ' '.join(words)


import torch
import random
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM

def get_synonyms_from_wordnet(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def replace_with_context(text, target_prob=0.2):
    tokenizer = BertTokenizer.from_pretrained('../models/bert-large')
    model = BertForMaskedLM.from_pretrained('../models/bert-large')

    words = text.split()
    num_words = len(words)
    replaceable_indices = []

    for i, word in enumerate(words):
        if get_synonyms_from_wordnet(word):
            replaceable_indices.append(i)

    num_to_replace = int(min(target_prob, len(replaceable_indices) / num_words) * num_words)
    indices_to_replace = random.sample(replaceable_indices, num_to_replace)

    real_replace = 0
    for i in indices_to_replace:
        original_word = words[i]

        # Create a sentence with a [MASK] token
        masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
        masked_text = " ".join(masked_sentence)
        
        # Use BERT to predict the token for [MASK]
        inputs = tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True)
        mask_position = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits[0, mask_position]
        predicted_indices = torch.argsort(predictions, descending=True)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
        words[i] = predicted_tokens[0]
        real_replace += 1

    print(f"Target Replace Rate: {target_prob}, Real Replace Rate: {real_replace / num_words}")
    return ' '.join(words)


    
    