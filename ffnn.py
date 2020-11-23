import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import trange

# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

unk = '<UNK>'

class FFNN(nn.Module):
	def __init__(self, input_dim, h, output_dim):
		super(FFNN, self).__init__()
		self.h = h
		self.W1 = nn.Linear(input_dim, h) 
		self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
		self.W2 = nn.Linear(h, output_dim)
    # The below two lines are not a source for an error
		self.softmax = nn.LogSoftmax(dim=1) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
		self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		# The z_i are just there to record intermediary computations for your clarity
		z1 = self.W1(input_vector).to(get_device())
		z2 = self.W2(self.activation(z1)).to(get_device()) 
		predicted_vector = self.softmax(z2)
		return predicted_vector
	
	def load_model(self, save_path):
		self.load_state_dict(torch.load(save_path))
	
	def save_model(self, save_path):
		torch.save(self.state_dict(), save_path)


def train_epoch(model, train_loader, optimizer, printing = True):
	model.train()
	total = 0
	loss = 0
	correct = 0
	total_loss = 0
	#for (input_batch, expected_out) in tqdm(train_loader, leave=False, desc="Training Batches"):
	for (input_batch, expected_out) in train_loader:
		optimizer.zero_grad() 
		# perform a prediciton using the current model weights
		output = model(input_batch.to(get_device()))
		total += output.size()[0]
		_, predicted = torch.max(output, 1)
		correct += (expected_out == predicted.to("cpu")).cpu().numpy().sum()
		# compute the prediction loss
		loss = model.compute_Loss(output, expected_out.to(get_device()))
		total_loss += loss.item()
		# perform backward propagation to compute the gradients
		loss.backward()
		# update the weights
		optimizer.step()

	# Print accuracy
	if printing:
		print('train accuracy',correct/total)
		print('train loss',total_loss/len(train_loader))
	return


def evaluation(model, val_loader, optimizer, printing = True):
	model.eval()
	loss = 0
	correct = 0
	total = 0
	total_loss = 0
	#for (input_batch, expected_out) in tqdm(val_loader, leave=False, desc="Validation Batches"):
	for (input_batch, expected_out) in val_loader:
		output = model(input_batch.to(get_device()))
		total += output.size()[0]
		_, predicted = torch.max(output, 1)
		correct += (expected_out.to("cpu") == predicted.to("cpu")).cpu().numpy().sum()

		loss += model.compute_Loss(output, expected_out.to(get_device()))
	loss /= len(val_loader)
 	# Print validation metrics
	if printing:
		print('val accuracy',correct/total)
		print('val loss', loss.item())
	return correct/total, loss.item()

def train_and_evaluate(number_of_epochs, model, train_loader, val_loader, save_file_name = None):
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	min_val_acc = 0
	for epoch in trange(number_of_epochs, desc="Epochs"):
		print('Epoch: ',epoch+1)
		train_epoch(model, train_loader, optimizer) 
		print()
		val_acc, val_loss = evaluation(model, val_loader, optimizer)
	
		if save_file_name and val_acc > min_val_acc:
				min_val_acc = val_acc
				model.save_model(save_file_name)
	return