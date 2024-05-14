import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets

	

if __name__ == '__main__':
	#加载配置文件
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	
	#读取配置文件
	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	#获取训练数据集和评估数据集
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	#初始化服务器对象
	server = Server(conf, eval_datasets)
	clients = []
	#为每个客户端创建一个模型
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n")
	print(f"客户端数为:{conf['no_models']}")
	for e in range(conf["global_epochs"]):
		#随机选择conf['k']个客户端
		candidates = random.sample(clients, conf["k"])
		for client in candidates:
			print(f"选择的客户端ID为:\t{client.client_id}")
		#初始化权重累计器
		weight_accumulator = {}
		#初始化全局模型的每一层参数,设置初始值全为0
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		#对于每一个被选中的客户端执行
		for c in candidates:
			#设置选中的客户端1为恶意客户端
			if c.client_id == 1:
				print("malicious client")
				#恶意客户端使用后门的方法进行本地训练
				diff = c.local_train_malicious(server.global_model)
			else:
				#正常的客户端使用正常的方法进行本地训练
				diff = c.local_train(server.global_model)
			#将每个客户端的参数更新累加
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		#服务器聚合全局模型的更新
		server.model_aggregate(weight_accumulator)
		#服务器使用评估数据集评估全局模型
		acc, loss = server.model_eval()
		#输出精度,损失
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
				
			
		
		
	
		
		
	