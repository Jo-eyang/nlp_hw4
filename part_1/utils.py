import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # QWERTY keyboard adjacent keys mapping
    qwerty_neighbors = {
        'a': ['s', 'q', 'w', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x'],
    }
    
    text = example["text"]
    words = word_tokenize(text)
    transformed_words = []
    
    for word in words:
        rand = random.random()
        
        if rand < 0.1:  # 10% chance for typo transformation
            word_chars = list(word)
            # Apply typos to characters with higher probability
            for i, char in enumerate(word_chars):
                lower_char = char.lower()
                if lower_char in qwerty_neighbors and random.random() < 0.15:  # 15% chance per character
                    neighbor = random.choice(qwerty_neighbors[lower_char])
                    # Preserve case
                    word_chars[i] = neighbor.upper() if char.isupper() else neighbor
            word = ''.join(word_chars)
        elif rand < 0.4:  # 30% chance for synonym replacement
            try:
                synsets = wordnet.synsets(word.lower())
                if synsets:
                    # Try multiple synsets to find a good synonym
                    for synset in synsets[:3]:
                        lemmas = synset.lemmas()
                        if len(lemmas) > 1:
                            synonym_candidates = [l.name() for l in lemmas if l.name() != word.lower()]
                            if synonym_candidates:
                                synonym = random.choice(synonym_candidates)
                                word = synonym.replace('_', ' ')
                                break
            except:
                pass
        elif rand < 0.45:  # 5% chance for character deletion/insertion
            if random.random() < 0.5:
                # Random character insertion
                if len(word) > 1 and random.random() < 0.4:
                    pos = random.randint(0, len(word) - 1)
                    char_to_insert = random.choice('aeiourlst')
                    word = word[:pos] + char_to_insert + word[pos:]
            else:
                # Random character deletion
                if len(word) > 2 and random.random() < 0.4:
                    pos = random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos+1:]
        
        transformed_words.append(word)
    
    # Reconstruct the text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
