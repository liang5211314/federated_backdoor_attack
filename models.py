
import torch 
from torchvision import models
import math

def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 
#用于计算两个模型之间的欧几里得距离,在后门攻击计算损失的时候使用
def model_norm(model_1, model_2):
	#初始化平方和0
	squared_sum = 0
	#遍历模型model_1中每一层参数及其名称
	for name, layer in model_1.named_parameters():
	#	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
		#计算当前层参数与模型model_2对应层数差的平方和
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)#返回所有层参数差的平方和的平方根,即欧几里得距离
