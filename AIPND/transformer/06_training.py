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

class TokenIdsDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size


    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, pos):
        assert pos < len(self.data) - self.block_size

        x = self.data[pos : pos + self.block_size]
        y = self.data[pos + 1 : pos + 1 + self.block_size]

        return x, y


#######################
# Config

config = {
    "vocabulary_size": tokenizer.vocabulary_size(),
    "context_size": 256,
    "embedding_dim": 768,
    "heads_num": 12,
    "layers_num": 10,
    "dropout_rate": 0.1,
    "use_bias": False,
}

config["head_size"] = config["embedding_dim"] // config["heads_num"]


#######################
# Attention Block

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["embedding_dim"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["embedding_dim"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["embedding_dim"], config["head_size"], bias=config["use_bias"])

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("causal_attention_mask", casual_attention_mask)


    def forward(self, input): # (B, C, embedding_din) B = batch size, C = context size
        bach_size, tokens_num, embedding_dim = input.shape
        Q = self.Q_weights(input) # (B, C, head_size)
        K = self.K_weights(input) # (B, C, head_size)
        V = self.V_weights(input) # (B, C, head_size)

        attention_scores = Q @ K.transpose(1, 2)  # (B, C, C)
        attention_scores = attention_scores.masked_fill(
            self.causal_attention_mask[:tokens_num, :tokens_num] == 0,
            -torch.inf
        )
        attention_scores = attention_scores / ( K.shape[-1] ** 0.5 )
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V # (B, C, head_size)

input = torch.rand(8, config["context_size"], config["embedding_dim"])
ah = AttentionHead(config)
output = ah(input)
print(output.shape) # Batche size, context size, head size

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        self.linear = nn.Linear(config["embedding_dim"], config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])


    def forward(self, input):
        # print(f"Input shape: {input.shape}")
        heads_outputs = [head(input) for head in self.heads]

        scores_change = torch.cat(heads_outputs, dim=-1)
        # print(f"heads shape: {scores_change.shape}")

        scores_change = self.linear(scores_change)
        return self.dropout(scores_change)

mha = MultiHeadAttention(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = mha(input)
print(output.shape) # Now the head size is equal to embedding_dim


#######################
# Transformer Block

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"] * 4, config["embedding_dim"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)

ff = FeedForward(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = ff(input)
print(output.shape) # Batche size, context size, dimensionality of embedding

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embedding_dim"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embedding_dim"])

    def forward(self, input):
        residual = input
        x = self.multi_head(self.layer_norm_1(input))
        x = x + residual

        residual = x
        x = self.feed_forward(self.layer_norm_2(x))
        return x + residual

tb = TransformerBlock(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = tb(input)
print(output.shape)  # Batche size, context size, dimensionality of embedding


#######################
# GPT Model

class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding_layer = nn.Embedding(
            config["vocabulary_size"], config["embedding_dim"]
        )
        self.positional_embedding_layer = nn.Embedding(
            config["context_size"], config["embedding_dim"]
        )

        blocks = [TransformerBlock(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)

        self.layer_norm = nn.LayerNorm(config["embedding_dim"])
        self.unembedding = nn.Linear(
            config["embedding_dim"], config["vocabulary_size"], bias=False
        )

    def forward(self, input_ids):
        batch_size, tokens_num = input_ids.shape

        x = self.token_embedding_layer(input_ids)
        sequence = torch.arange(tokens_num, device=device)
        x = x + self.positional_embedding_layer(sequence)

        x = self.layers(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)

        return x

model = DemoGPT(config).to(device)
output = model(tokenizer.encode("Hi").unsqueeze(dim=0).to(device))  # We need to add an extra dimension because encode() returns a 1 dimensional vector but our model expects a batch of inputs. We use unsqueeze() to add an extra dimension.
print(output.shape)  # Batch size, input tokens, vocabulary size

def generate(model, prompt_ids, max_tokens):
    output_ids = prompt_ids
    for _ in range(max_tokens):
        if output_ids.shape[1] >= config["context_size"]:
            break
        with torch.no_grad():
            logits = model(output_ids)

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # Sample a random token given the softmax distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
    return output_ids

def generate_with_prompt(model, tokenizer, prompt, max_tokens=100):
    model.eval()  # Set the model to inference mode

    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to(device)

    return tokenizer.decode(generate(model, prompt, max_tokens=max_tokens)[0])

next_word = generate_with_prompt(model, tokenizer, "First Citizen:\n")
print(next_word)  # outputs gibberish because we have not trained the model yet


##############################
# Training without Validation

batch_size = 64

train_iterations = 5000
evaluation_interval = 100
learning_rate = 4e-4

train_data = tokenizer.encode(text).to(device)  # One dimensional tensor representing the token ids
train_dataset = TokenIdsDataset(train_data, config["context_size"])

train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for step_num, sample in enumerate(train_dataloader):

    model.train()
    input, targets = sample
    input, targets = input.to(device), targets.to(device)
    logits = model(input)

    logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
    targets_view = targets.view(batch_size * config["context_size"])

    loss = F.cross_entropy(logits_view, targets_view)
    # Backward propagation
    loss.backward()
    # Update model parameters
    optimizer.step()
    # Set to None to reduce memory usage
    optimizer.zero_grad(set_to_none=True)

    print(f"Step {step_num}: loss = {loss.item():.3f}")

    if step_num % evaluation_interval == 0:
        print("DemoGPT\n" + generate_with_prompt(model, tokenizer, "\n"))
