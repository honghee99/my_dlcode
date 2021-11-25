# 模块加载 ##################################################
from __future__ import print_function, division    # from __future__ import是为了解决python个版本之间的差异
                                         # print_function针对输出问题比如Python3中print()需要括号，而python2中print不需要括号
                                         # division针对除法问题，/为精确除法，//为圆整除法
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler    # 学习速率衰减
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy           # copy模块，用于后面模型参数的深拷贝  copy.deepcopy
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.ion()              # interactive mode 启动matplotlib是python中的库中的画图交互模式
# matplotlib是python中的可视化库，其显示模式默认为阻塞模式（block mode），也就是说plt.show()后程序化挂起，不载继续往下执行，如果想要继续往下执行就需要关闭图片显示。
# 如果想要显示多张图片，就需要使用plt.ion() ，将阻塞模式切换为交互模式（interactive mode），这样，即使在脚本中遇到plt.show()，代码还是会继续执行。

# 数据加载 ##################################################
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([               # 将所有的transform操作合并在一起执行
        transforms.RandomResizedCrop(224),      # 将图片随机裁剪后resize到同一个size中
        transforms.RandomHorizontalFlip(),      # 依据概率p对PIL图片进行水平翻转，默认概率p为0.5
        transforms.ToTensor(),                  # 将一个image转换为【C,H,W】的tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # Normalize(mean, std)给定均值和方差，对图像进行标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = r'/data/lihongxi/learn_and_lab/hymenoptera_data'    # 样本地址

# 构建训练和验证的样本数据集，字典格式
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),    # os.path.join实现路径拼接
                                          data_transforms[x])           # data_transforms也是字典格式，
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，还可以针对minibatch、多线程等进行针对性构建
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,    #
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# 粉笔计算训练与测试的样本数，字典格式
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}   # 训练与测试的样本数
class_names = image_datasets['train'].classes           # 样本的类别，分别对应着蜜蜂和蚂蚁

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # 判断是否使用cpu
print(dataset_sizes,class_names)
# 图片显示函数 ##################################################
# 这个函数主要用于将图片打印到输出栏中
def imshow(inp, title=None):
    """Imshow for Tensor."""
    print(inp.shape) # 辅助语句               # 原始的inp是tensor格式，shape是【C,H,W】torch.Size([3, 228, 906])    PS：
    inp = inp.numpy().transpose((1, 2, 0))   # 将tensor格式转换为numpy格式的三维矩阵，并且transpose为【W,H,C】的格式以便于plt输出。
    print(inp.shape)# 辅助语句
    mean = np.array([0.485, 0.456, 0.406])   # 均值
    std = np.array([0.229, 0.224, 0.225])    # 标准化
    inp = std * inp + mean                   # 将图像反标准化为原来的样子
    inp = np.clip(inp, 0, 1)                 # 只对inp中所有的元素值进行裁剪，按照最小值0，最大值为1进行裁剪,以免溢出RGB值的上下限0-255
    print(inp.shape) # 辅助语句
    plt.imshow(inp)                           # 图像输出
    if title is not None:                   # 如果输出图像的标题不为空
        plt.title(title)                      # 对plt输出的结果添加title
    plt.pause(0.001)                          # pause a bit so that plots are updated，常用于动图的刷显



# Get a batch of training data  获取一个batch的样本
inputs, classes = next(iter(dataloaders['train']))   # iter用于生成迭代器，next用于返回迭代器的下一个条目

# Make a grid from batch  将batch中的4章图片拼成一整张
out = torchvision.utils.make_grid(inputs)    # make_grid的作用是将若干张图片拼成一张

imshow(out, title=[class_names[x] for x in classes])    # 调用上面定义的imshow函数
# 模型训练的函数 ##################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):   # 括号中的参数是模型，损失函数标准，优化器，学习速率衰减方式，训练次数
    since = time.time()     # 开始时间

    best_model_wts = copy.deepcopy(model.state_dict())    # 先深拷贝一份当前模型的参数（wts=weights），后面迭代过程中若遇到更优模型则替换
    best_acc = 0.0                                        # 最佳正确率，用于判断是否替换best_model_wts

    for epoch in range(num_epochs):      # 开启每一个epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))   # Epoch 20/24，从epoch 0开始，因此num_epochs - 1
        print('-' * 10)                  # 在输出栏画出一条分割线 ----------

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:   # 每个epoch中都包含训练与验证两个阶段
            if phase == 'train':         # 训练阶段
                model.train()            # Set model to training mode
            else:                        # 测试阶段
                model.eval()             # Set model to evaluate mode
                # 与train不同的是，test过程中没有batch-normalization与dropout，因此要区别对待。
                # batchnorm是针对minibatch的，测试过程中每个样本单独测试，不存在minibatch

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.   # 每个阶段都需要遍历所有的样本
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)           # 将inputs所指向的样本都copy一份发哦device所指向的GPU中，tensor与numpy都是矩阵，但是前者可以在GPU上运行，后者不可以
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()    # 将梯度重新置为0
                                         # pytorch中，backward函数中所有的参数梯度是被累加的（Variable.grad=Variable.grad+new_grad），
                                         # 而非替换的，因此在每次开启一个batch的训练中需要将梯度重新置为0
                                         # PS：对于那种计算机性能不强，但是想要设置较大的batch-size的训练情况，可以设置为训练多个batch后再置一次零

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):   # torch.set_grad_enabled(False/True)是上下文管理器，用于确定是否对with下的所有语句设计的参数求导，如果设置为false则新节点不可求导
                    outputs = model(inputs)            # 网络模型的前向传播，就是为了从输入得到输出
                    _, preds = torch.max(outputs, 1)   # 在维度1（行方向）查找最大值
                    loss = criterion(outputs, labels)  # 输出结果与label相比较

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()     # 误差反向传播，计算每个w与b的更新值
                        optimizer.step()    # 将这些更新值施加到模型上

                # statistics
                running_loss += loss.item() * inputs.size(0)         # 计算当前epoch过程中，所有batch的损失和
                running_corrects += torch.sum(preds == labels.data)  # 判断正确的样本数
            if phase == 'train':    # 完成本次epoch所有样本的训练与验证之后，就对学习速率进行修正
                scheduler.step()     # 在训练过程中，要根据损失的情况修正学习速率

            epoch_loss = running_loss / dataset_sizes[phase]               # 当前epoch的损失值是loss总和除以样本数
            epoch_acc = running_corrects.double() / dataset_sizes[phase]   # 当前epoch的正确率

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(         # 输出train/test，损失、正确率
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:                # 如果是val阶段，并且当前epoch的acc比best acc大
                best_acc = epoch_acc                                    # 就替换best acc为当前epoch的acc
                best_model_wts = copy.deepcopy(model.state_dict())      # 将best_model替换为当前模型

        print()     # 输出空格

    time_elapsed = time.time() - since                           # 结束时间减去开始时间是所有epoch训练完成后的训练耗时
    print('Training complete in {:.0f}m {:.0f}s'.format(         # Training complete in 30m 7s  输出总耗时
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))                # 输出验证正确率 Best val Acc: 0.954248

    # load best model weights
    model.load_state_dict(best_model_wts)                        # 将最佳模型的相关参数加载到model中
    return model
# 显示模型的预测输出的函数 ##################################################
def visualize_model(model, num_images=6):
    was_training = model.training     # 检查是否处于训练模式
    model.eval()                      # 调用测试方法
    images_so_far = 0                 # 到目前为止的验证集图片数，用于绘制subplot
    fig = plt.figure()                # plt画图句柄

    with torch.no_grad():                   # 在该with结构下，参数不会进行梯度计算与相关的更新
        for i, (inputs, labels) in enumerate(dataloaders['val']):       # 验证模式
            inputs = inputs.to(device)            # 将所有验证样本拷贝到GPU
            labels = labels.to(device)

            outputs = model(inputs)               # 模型的前向传播
            _, preds = torch.max(outputs, 1)      # preds是模型的预测标签

            for j in range(inputs.size()[0]):    # 遍历所有验证样本
                images_so_far += 1                # 当前图片位置 +1
                ax = plt.subplot(num_images//2, 2, images_so_far)    # subplot(m,n,j)表示将窗口分为m行n列，当前的位置时j
                ax.axis('off')                    # 不显示轴线与刻度
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))   # 图片的title
                imshow(inputs.cpu().data[j])      # 调用imshow函数，进行画图

                if images_so_far == num_images:      # 如果当前样本位置==样本图片综述
                    model.train(mode=was_training)   # # 在测试之后将模型恢复之前的形式
                    return
        model.train(mode=was_training)    # 在测试之后将模型恢复之前的形式
#  第一种迁移模式：Finetuning the convnet（微调整个模型）
# 第一种迁移学习方法：加载预训练模型然后充值最后几个全连接层，并且对多有层都进行反向微调
# 首先，在网站上下载已经预训练好的模型参数，然后定义好每一个续联OP



# ## 迁移方法 1： 对模型所有层的所有参数都进行目标域的训练，
#
# model_ft = models.resnet18(pretrained=True)    # 加载resnet18这个模型，pretrained=True表示还要加载预训练好的参数
# num_ftrs = model_ft.fc.in_features             # 全连接层的输入的特征数
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)           # 利用线性映射将原来的num_ftrs转换为2（蚂蚁和蜜蜂）
#                                                # 将最后一个全连接由（512， 1000）改为(512, 2)   因为原网络是在1000类的ImageNet数据集上训练的
# model_ft = model_ft.to(device)                 # 设置计算采用的设备，GPU还是CPU
#
# criterion = nn.CrossEntropyLoss()              # 交叉熵损失函数
#
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)   # 优化器，对加载的模型中所有参数都进行优化
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)   # 学习速率衰减速度这只
#
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)    # 模型训练
# # 在CPU上完成训练大约需要15-25分钟，在GPU上则一分钟不到





## 迁移方法 2： 底部的卷积层全部冻结，在目标域仅仅对顶部的全连接层进行训练

model_conv = torchvision.models.resnet18(pretrained=True)      # 加载模型
for param in model_conv.parameters():    # 依次遍历所有参数
    param.requires_grad = False          # 通过requires_grad == False的方式来冻结这些层的参数

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features       # 全连接层的输入特征维数
model_conv.fc = nn.Linear(num_ftrs, 2)     # 通过线性变换将维数从原来resnet18的1000维（imagenet的1000的类别）降低到2维

model_conv = model_conv.to(device)         # 判断使用GPU或者CPU

criterion = nn.CrossEntropyLoss()          # 以交叉熵损失函数作为标准

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)    # 仅仅对最后一层进行优化

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)    # 学习速率衰减

model_conv = train_model(model_conv, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=25)
# 相比于第一种迁移学习方法，第二种方法所需训练用时大约只有前者的一半

visualize_model(model_conv)      # 调用子函数输出模型预测结果

plt.ioff()       # 前面利用ion()命令开启了交互模式，如果没有使用ioff()关闭的话，输出图像会一闪而过。要想防止这种情况，需要在plt.show()之前加上ioff()命令。
plt.show()

