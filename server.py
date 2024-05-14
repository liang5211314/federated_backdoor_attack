
import models, torch

#定义服务器类
class Server(object):
	#初始化方法
	def __init__(self, conf, eval_dataset):
		#初始化配置信息
		self.conf = conf 
		#获取全局模型
		self.global_model = models.get_model(self.conf["model_name"]) 
		#创建评估数据的加载器
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	#模型参数聚合方法
	def model_aggregate(self, weight_accumulator):
		#遍历全局模型的每一层参数
		for name, data in self.global_model.state_dict().items():
			#计算每层参数的更新值,乘以lambda系数
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			#确保更新值和模型参数值的类型一直,然后进行更新参数操作
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)

	#模型评估方法
	def model_eval(self):
		#设置全局模型为评估模式
		self.global_model.eval()
		#初始化总损失和正确的预测数
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		#遍历评估数据的加载器
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			#更新数据集的大小
			dataset_size += data.size()[0]
			#如果cuda可用,将数据和标签移动到GPU进行计算
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			#获取到输出
			output = self.global_model(data)
			#计算交叉熵损失,并累加到总损失
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			#获取预测正确的标签索引
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			#更新预测正确的标签索引的数量
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
		#计算精确率和平均损失
		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size
		#返回评估的准确率和平均损失
		return acc, total_l