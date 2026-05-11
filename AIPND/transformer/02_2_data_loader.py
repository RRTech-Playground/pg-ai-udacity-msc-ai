import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, RandomSampler

from pathlib import Path


#######################
# Device

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Device set to: {device}")


#######################
# Data

text = Path("data/tiny-shakespeare.txt").read_text()
print(text[0:100])

class CharTokenizer:
    def __init__(self, vocabulary):
        self.token_id_for_char = {
            char: token_id for token_id, char in enumerate(vocabulary)
        }
        self.char_for_token_id = {
            token_id: char for token_id, char in enumerate(vocabulary)
        }

    @staticmethod
    def train_from_text(text):
        vocabulary = set(text)
        return CharTokenizer(sorted(list(vocabulary)))

    def encode(self, text):
        token_ids = []
        for char in text:
            token_ids.append(self.token_id_for_char[char])
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        chars = []
        for token_id in token_ids.tolist():
            chars.append(self.char_for_token_id[token_id])
        return "".join(chars)

    def vocabulary_size(self):
        return len(self.token_id_for_char)

tokenizer = CharTokenizer.train_from_text(text)
print(tokenizer.encode("Hello world"))
print(tokenizer.decode(tokenizer.encode("Hello world")))
print(f"Vocabulary size: {tokenizer.vocabulary_size()}")


# Step 1 - Define the `TokenIdsDataset` Class
class TokenIdsDataset(Dataset):
    def __init__(self, data, block_size):
        # Save data and block size
        self.data = data
        self.block_size = block_size


    def __len__(self):
        # If every position can be a start of an item,
        # and all items should be "block_size", compute the size
        # of the dataset
        return len(self.data) - self.block_size

    def __getitem__(self, pos):
        # Check if the input position is valid
        assert pos < len(self.data) - self.block_size

        # Get an item from position "pos"
        x = self.data[pos : pos + self.block_size]

        # Get a target item (shifted by one position)
        y = self.data[pos + 1 : pos + 1 + self.block_size]

        # Return both
        return x, y


# Step 2 - Tokenize the Text

# Encode text using the tokenizer
# Create "TokenIdsDataset" with the tokenized text, and block_size=64
tokenized_text = tokenizer.encode(text)
dataset = TokenIdsDataset(tokenized_text, block_size=64)


# Step 3 - Retrieve the First Item from the Dataset

# Get the first item from the dataset
# Decode "x" using tokenizer.decode
x, y = dataset[0]
print(x)
print(y)
#print(f"First item (x): {tokenizer.decode(x)}")
#print(f"Target item (y): {tokenizer.decode(y)}")

# RandomSampler allows to read random items from a datasset
sampler = RandomSampler(dataset, replacement=True)
# Dataloader will laod two random samplers using the sampler
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)


# Step 4 - Use a DataLoader

# Get a single batch from the "dataloader"
# For this call the `iter` function, and pass DataLoader instance to it. This will create an iterator
# Then call the `next` function and pass the iterator to it to get the first training batch
x, y = next(iter(dataloader))
print(x.shape)
print(x)

# Decode input item
print(tokenizer.decode(x[0]))

# Decode target item
print(tokenizer.decode(y[0]))
