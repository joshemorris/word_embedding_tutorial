# this is from the PyTorch tutorial on word embeddings found here:
# http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py
# Original Author: Robert Guthrie
# Example completed by: Josh Morris

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2 # to left and right
EMBEDDING_DIM = 10

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

#use set to create vocab with no duplicates
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
	context = [raw_text[i - 2], raw_text[i - 1],
						raw_text[i + 1], raw_text[i + 2]]
	target = raw_text[i]
	data.append((context, target))

class CBOW(nn.Module):

	def __init__(self, vocab_size, embedding_dim, context_size):
		super(CBOW, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * 2 * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1,-1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out)
		return log_probs

losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(100):
	total_loss = torch.Tensor([0])
	for context, target in data:

		#turn context into indices and wrap in var
		context_idxs = [word_to_ix[w] for w in context]
		context_var = autograd.Variable(torch.LongTensor(context_idxs))

		#clear out gradients from previous instance
		model.zero_grad()

		#perform forward pass
		log_probs = model(context_var)

		#compute loss
		loss = loss_function(log_probs, autograd.Variable(
			torch.LongTensor([word_to_ix[target]])))

		#backward pass and update gradient
		loss.backward()
		optimizer.step()

		total_loss += loss.data

	losses.append(total_loss)

print(losses)