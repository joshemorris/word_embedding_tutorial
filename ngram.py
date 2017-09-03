# this is from the PyTorch tutorial on word embeddings found here:
# http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py
# Original Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

#Shakespeare Sonnet 2

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# build list of tuples
# ([word_i-2, word_i-1], target_word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
						for i in range(len(test_sentence) - 2)]

print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

	def __init__(self, vocab_size, embedding_dim, context_size):
		super(NGramLanguageModeler, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1, -1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out)
		return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(100): 
	total_loss = torch.Tensor([0])
	for context, target in trigrams:

		# turn words into int indices and wrap in variables
		context_idxs = [word_to_ix[w] for w in context]
		context_var = autograd.Variable(torch.LongTensor(context_idxs))

		#zero out gradients from previous instance
		model.zero_grad()

		#perform forward pass
		log_probs = model(context_var)

		#compute loss function
		loss = loss_function(log_probs, autograd.Variable(
			torch.LongTensor([word_to_ix[target]])))

		#backward pass and update gradient
		loss.backward()
		optimizer.step()

		total_loss += loss.data
	losses.append(total_loss)

print(losses)
