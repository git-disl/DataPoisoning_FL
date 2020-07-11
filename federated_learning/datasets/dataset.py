from abc import abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import numpy

class Dataset:

	def __init__(self, args):
		self.args = args

		self.train_dataset = self.load_train_dataset()
		self.test_dataset = self.load_test_dataset()

	def get_args(self):
		"""
		Returns the arguments.

		:return: Arguments
		"""
		return self.args

	def get_train_dataset(self):
		"""
		Returns the train dataset.

		:return: tuple
		"""
		return self.train_dataset

	def get_test_dataset(self):
		"""
		Returns the test dataset.

		:return: tuple
		"""
		return self.test_dataset

	@abstractmethod
	def load_train_dataset(self):
		"""
		Loads & returns the training dataset.

		:return: tuple
		"""
		raise NotImplementedError("load_train_dataset() isn't implemented")

	@abstractmethod
	def load_test_dataset(self):
		"""
		Loads & returns the test dataset.

		:return: tuple
		"""
		raise NotImplementedError("load_test_dataset() isn't implemented")

	def get_train_loader(self, batch_size, **kwargs):
		"""
		Return the data loader for the train dataset.

		:param batch_size: batch size of data loader
		:type batch_size: int
		:return: torch.utils.data.DataLoader
		"""
		return Dataset.get_data_loader_from_data(batch_size, self.train_dataset[0], self.train_dataset[1], **kwargs)

	def get_test_loader(self, batch_size, **kwargs):
		"""
		Return the data loader for the test dataset.

		:param batch_size: batch size of data loader
		:type batch_size: int
		:return: torch.utils.data.DataLoader
		"""
		return Dataset.get_data_loader_from_data(batch_size, self.test_dataset[0], self.test_dataset[1], **kwargs)

	@staticmethod
	def get_data_loader_from_data(batch_size, X, Y, **kwargs):
		"""
		Get a data loader created from a given set of data.

		:param batch_size: batch size of data loader
		:type batch_size: int
		:param X: data features
		:type X: numpy.Array()
		:param Y: data labels
		:type Y: numpy.Array()
		:return: torch.utils.data.DataLoader
		"""
		X_torch = torch.from_numpy(X).float()

		if "classification_problem" in kwargs and kwargs["classification_problem"] == False:
			Y_torch = torch.from_numpy(Y).float()
		else:
			Y_torch = torch.from_numpy(Y).long()
		dataset = TensorDataset(X_torch, Y_torch)

		kwargs.pop("classification_problem", None)

		return DataLoader(dataset, batch_size=batch_size, **kwargs)

	@staticmethod
	def get_tuple_from_data_loader(data_loader):
		"""
		Get a tuple representation of the data stored in a data loader.

		:param data_loader: data loader to get data from
		:type data_loader: torch.utils.data.DataLoader
		:return: tuple
		"""
		return (next(iter(data_loader))[0].numpy(), next(iter(data_loader))[1].numpy())
