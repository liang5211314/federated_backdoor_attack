
import models, torch, copy
import numpy as np
import matplotlib.pyplot as plt
#定义一个客户端的类
class Client(object):
	#初始化方法
	def __init__(self, conf, model, train_dataset, id = -1):
		#配置信息
		self.conf = conf
		#获取本地模型参数
		self.local_model = models.get_model(self.conf["model_name"]) 
		#客户端ID
		self.client_id = id
		#训练数据集
		self.train_dataset = train_dataset
		#根据客户端ID和模型数量划分数据集
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]
		#创建数据加载器,使用指定批次大小和采样器
		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
		
			
	#进行本地训练的方法(不包含后门攻击)
	def local_train(self, model):
		#将全局模型参数复制到本地模型中
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		#创建优化器SGD
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		#训练本地模型
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				#如果GPU可用,则将数据移动到GPU计算
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				#梯度归零,前向传播,计算损失,反向传播,更新参数
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
			print("正常训练 Epoch %d done." % e)
		#返回本地模型与全局模型参数的差异
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			
		return diff
	#恶意客户端训练的方法,包含后门攻击
	def local_train_malicious(self, model):
		                                                            #将全局模型参数复制到本地模型
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
		#创建优化器
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		#定义后门中毒出发的位置
		pos = []
		for i in range(2, 28):
			pos.append([i, 3])
			pos.append([i, 4])
			pos.append([i, 5])
		#训练本地模型,并在每个批次中对特定样本实施后门攻击
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				#对每个被选中的样本实施后门攻击
				for k in range(self.conf["poisoning_per_batch"]):
					img = data[k].numpy()
					for i in range(0,len(pos)):
						img[0][pos[i][0]][pos[i][1]] = 1.0
						img[1][pos[i][0]][pos[i][1]] = 0
						img[2][pos[i][0]][pos[i][1]] = 0
					#将被攻击的样本的标签
					target[k] = self.conf['poison_label']
				# for k in range(32):
				# 		img = data[k].numpy()
				#
				# 		img = np.transpose(img, (1, 2, 0))
				# 		plt.imshow(img)
				# 		plt.show()
				#如果cuda可用,将数据移步到GPU
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				#梯度归零,前向传播,计算损失(包括类别损失和分布损失)
				optimizer.zero_grad()
				output = self.local_model(data)
				
				class_loss = torch.nn.functional.cross_entropy(output, target)
				dist_loss = models.model_norm(self.local_model, model)
				loss = self.conf["alpha"]*class_loss + (1-self.conf["alpha"])*dist_loss
				loss.backward()
			
				optimizer.step()
			print("后门攻击 Epoch %d done." % e)
		#返回更新后的本地模型参数
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = self.conf["eta"]*(data - model.state_dict()[name])+model.state_dict()[name]
			
		return diff		
		