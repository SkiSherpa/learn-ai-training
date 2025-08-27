import tiktoken

enc = tiktoken.get_encoding('gpt2')
# print(enc.n_vocab, "prints num of tokens gpt2 has") # 50257
# so when we print the vector "hii there", we get only 3 nums.
# its how they encoded the sub-words, instead of by letter -> num.
# print(enc.encode("hii there"))

###### ---------- VIDEO EXAMPLE ---------- #######

######## Video Example, just does a typical mapping of char > num ########
# It's easier to follow with the simple mapping for learning
# have a Little Shakespear txt file to use (supply training?)

# Read the txt file
with open('ai-training-input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("size of map", vocab_size)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("Test it out")
print(encode("hii there"))
print(decode(encode("hii there")))

# we can tokenize the little Shakespear library
import torch 
# glosses over what the torch tensor is.
# Tensor - a way of storing vectors in a multidimensional system
# Where the dimensions are set to how the ai wants to think in
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000]) # the 1st 1,000 chars in the txt file

#### Getting a warning with numpy when I run it - claude says its b/c, NumPy provides better integration between PyTorch tensors and NumPy arrays - warning can be ignored at this point ####

############ VALIDATION SPLIT ###############
# Let's split the data into the first 90% of the data and save the last 10% for the validation data
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# usually you want to train it on chunks of the data, in this video we use the term "block_size"
block_size = 8 # 8 chars
print("9 nums vect", train_data[:block_size+1]) # give an array of 9 chars
# tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])

# the ai gets trained on the whole list of nums
# 47 follows 18, 56 follows 18,47, and so on 

# its trained on all the nums/chars from 1 to block_size
# we want it to be trained on a the smallest context to the largest

# Batches - a batch makes up many chunks of tensors
# Batches are used b/c GPU's can process in parallel for efficiency

torch.manual_seed(1337) # this keep the "random" nums seen in the video the same as the ones use here

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # torch.stack tf the 1x8 array to be a 8x1 col, but still an array
    x = torch.stack([data[i:i+block_size] for i in ix]) # getting chars from i to block size
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # the same but offset by 1
    # x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print('this is the val nums, the targets: ')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension - where in sequence
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when IP is {context.tolist()} the target: {target}")

# We now have our batch of IPs we want to feed into a transformer
print(xb)
# tensor([[24, 43, 58,  5, 57,  1, 46, 43],
#         [44, 53, 56,  1, 58, 46, 39, 58],
#         [52, 58,  1, 58, 46, 39, 58,  1],
#         [25, 17, 27, 10,  0, 21,  1, 54]])

##### CAN NOW ADD TO NEURAL NETWORKS ##########
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     loss = F.cross_entropy(logits, targets)

        # return logits, loss
        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)


