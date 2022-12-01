import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from DataTransform import DataLoad
from pathlib import Path
from accelerate import Accelerator
from transformers import AdamW, get_scheduler
import tqdm
from collections import OrderedDict
from fc_model import fc_model


"""
Training of the Breast Cancer detector
"""

class train():


	def __init__(self, train_dataloader, eval_dataloader, model_path):

		self.train_dataloader = train_dataloader
		self.eval_dataloader = eval_dataloader
		self.model_path = model_path

		# Accelerate training
		accelerator = Accelerator()
		
		# Loading the checkpoints of Med3D Segmentation Backbone
		checkpoint = torch.load(self.model_path, map_location = "cpu")
		Model = fc_model()
		model = Model.state_dict(checkpoint)

		optimizer = AdamW(Model.parameters(), lr=3e-5)

		self.train_dataloader, self.eval_dataloader, model, optimizer = accelerator.prepare(
			self.train_dataloader, self.eval_dataloader, model, optimizer)

		num_epochs = 5
		num_training_steps = num_epochs * len(self.train_dataloader)
		lr_scheduler = get_scheduler(
			"linear",
			optimizer=optimizer,
			num_warmup_steps=0,
			num_training_steps=num_training_steps)

		#progress_bar = tqdm(range(num_training_steps))


		model.train()
		for epoch in range(num_epochs):
			for batch in train_dataloader:
				outputs = model(**batch)
				loss = outputs.loss()
				accelerator.backward(loss)

				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				#progress_bar.update(1)
				


if __name__== "__main__" :
    
    path = "/home/alchimiste-12/Bureau/Med3D/pretrain/resnet_101.pth"
    dataloader = DataLoad()
    #trainloader = dataloader['train']
    #valloader = dataloader['val']
    dataloader2 = dataloader.DataTransforms()
    
    Train = train(dataloader2['train'], dataloader2['val'], path)
    

