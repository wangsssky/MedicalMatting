from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from sklearn.model_selection import KFold
from dataloader.data_loader import AlphaDataset


class AlphaDatasetSpliter():
	def __init__(self, opt, input_size):
		self.opt = opt
		self.train_dataset = AlphaDataset(
			dataset_location=opt.DATA_PATH, input_size=input_size)
		self.test_dataset = AlphaDataset(
			dataset_location=opt.DATA_PATH, input_size=input_size)
		self.kf = KFold(n_splits=opt.KFOLD, shuffle=False)

		self.splits = []

		if opt.DATASET == 'lidc':
			uid_dict = {}
			for idx, uid in enumerate(self.train_dataset.series_uid):
				pid = uid.split('_')[0]
				if pid in uid_dict.keys():
					uid_dict[pid].append(idx)
				else:
					uid_dict[pid] = [idx]

			pids = list(uid_dict.keys())
			np.random.seed(opt.RANDOM_SEED)
			np.random.shuffle(pids)
			for (train_pid_index, test_pid_index) in self.kf.split(np.arange(len(pids))):
				train_index = []
				test_index = []
				for pid_idx in train_pid_index:
					train_index += uid_dict[pids[pid_idx]]
				for pid_idx in test_pid_index:
					test_index += uid_dict[pids[pid_idx]]
				self.splits.append({'train_index': train_index, 'test_index': test_index})
		else:
			indices = list(range(len(self.train_dataset)))
			np.random.seed(opt.RANDOM_SEED)
			np.random.shuffle(indices)
			for (train_index, test_index) in self.kf.split(np.arange(len(self.train_dataset))):
				self.splits.append({
					'train_index': [indices[i] for i in train_index.tolist()],
					'test_index': [indices[i] for i in test_index.tolist()]})

	def get_datasets(self, fold_idx):
		train_indices = self.splits[fold_idx]['train_index']
		test_indices = self.splits[fold_idx]['test_index']
		train_sampler = SubsetRandomSampler(train_indices)
		test_sampler = SubsetRandomSampler(test_indices)
		train_loader = DataLoader(
			self.train_dataset, batch_size=self.opt.TRAIN_BATCHSIZE, sampler=train_sampler)
		test_loader = DataLoader(
			self.test_dataset, batch_size=self.opt.VAL_BATCHSIZE, sampler=test_sampler)
		print("Number of training/test patches:", (len(train_indices), len(test_indices)))

		return train_loader, test_loader
