from __future__ import division
import torch
import glob
import re
import os
from torchvision.utils import save_image
import torch.optim as optim
import statistics
import hydra
from part2_model.loss.triplet_loss_layers import loss_HardNet
import  configs.config_loader as cfg_loader
# 定义自定义的损失函数 HardNetLoss
class HardNetLoss(torch.nn.Module):
    def __init__(self, batch_size, loss_weight=0.1, margin=0.2):
        torch.nn.Module.__init__(self)

        # construct anchor, positive, and negative ids
        # 构建锚点、正样本和负样本的索引
        self.anchor_ids = range(0, batch_size * 3, 3)
        self.positive_ids = range(1, batch_size * 3, 3)
        self.negative_ids = range(2, batch_size * 3, 3)

        self.nb_anchor_samples = len(self.anchor_ids)
        self.nb_positive_samples = len(self.positive_ids)
        self.nb_negative_samples = len(self.negative_ids)
        self.weight = loss_weight
        self.margin = margin
        self.labels_mini_batch = None
        self.sizeMiniBatch = len(self.anchor_ids)

        assert (
            self.nb_anchor_samples == self.nb_positive_samples
            and self.nb_positive_samples == self.nb_negative_samples
        )

    def get_weight(self):
        return self.weight

    def set_actual_labels(self, new_label):
        self.labels_mini_batch = new_label
        return

    def get_label_batch(label_data, batch_size, batch_index):

        nrof_examples = np.size(label_data, 0)
        j = batch_index * batch_size % nrof_examples

        if j + batch_size <= nrof_examples:
            batch = label_data[j : j + batch_size]
        else:
            x1 = label_data[j:nrof_examples]
            x2 = label_data[0 : nrof_examples - j]
            batch = np.vstack([x1, x2])
        batch_int = batch.astype(np.int64)

        return batch_int

    def forward(self, input):
        # anchors = input[self.anchor_ids, :].cuda()
        # positives = input[self.positive_ids, :].cuda()
        anchors = input[self.anchor_ids, :]
        positives = input[self.positive_ids, :]
        pos, min_neg, loss = loss_HardNet(
            anchors,
            positives,
            margin=self.margin,
            batch_reduce="min",
            loss_type="triplet_margin",
        )
        return pos, min_neg, loss
# 工厂函数，用于创建不同类型的损失函数
def loss_factory(loss_id, batch_size, margin_value, loss_weight):
    assert loss_id == "HardNetLoss"
    if loss_id == "HardNetLoss":
        loss_layer = HardNetLoss(
            batch_size=batch_size,
            margin=margin_value,
            loss_weight=loss_weight,
        )

    return loss_layer


# 定义训练器
class Trainer(object):
    # 设置损失函数 self.criterion 为二进制交叉熵 (Binary Cross Entropy)。
    # 创建优化器 self.optimizer，使用 Adam 优化器，学习率设置为 0.003，优化的参数是模型的参数。
    # 设置训练的总周期数 (self.epochs) 为 200。
    # 如果启用了 self.args.resume ，则尝试加载之前训练的模型权重和优化器状态。
    def __init__(self,train_loader,model,opt):

        self.args = opt
        self.train_loader = train_loader
        self.model = model
        args = cfg_loader.get_config()
        self.batch_size = args.batch_size
        self.margin_value = args.margin_value,
        self.loss_weight = args.loss_weight,
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.image_size = args.image_size
        self.modelname = args.modelname

        # 使用工厂函数创建损失函数
        self.criterion = loss_factory(
            self.args.loss_layer,
            batch_size=self.batch_size,
            # 不知道为什么这里赋值之后类型就变tuple了, 干脆直接用args了
            margin_value=args.margin_value,
            loss_weight=self.loss_weight,
        )

        # 定义优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.epochs = args.nb_epoch
        # self.model.cuda()

        self.start_epoch = 0

        # 如果需要恢复训练，则加载已保存的模型权重和优化器状态
        if self.args.resume:
            framelist = glob.glob("trained_models" + '/**tar')
            numbers = []
            for k in range(len(framelist)):
                tmp = re.findall('\d+', framelist[k])
                num = int(tmp[0])
                numbers.append(num)
            self.start_epoch = max(numbers)

            modelfile = "unet-" + str(self.start_epoch) + ".pth.tar"

            if os.path.isfile("trained_models" + "/" + modelfile):
                print("load checkpoint from '{}'".format("trained_models"+"/"+modelfile))
                self.checkpoint = torch.load("trained_models"+"/"+modelfile)
                self.model.load_state_dict(self.checkpoint['state_dict'])
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                print("loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.checkpoint['epoch']))
                self.start_epoch += 1

    # 外层循环迭代训练的周期（epoch），从 self.start_epoch 开始，一直到 self.epochs - 1。
    # 内层循环迭代训练数据加载器 (self.train_loader) 中的每个批次数据。
    # 然后，通过将优化器的梯度置零 (self.optimizer.zero_grad())，计算模型的输出 (output)，并计算损失值 (loss)，
    # 这里使用了二进制交叉熵损失函数。
    # 使用反向传播 (loss.backward()) 来计算梯度，并使用优化器来更新模型的权重 (self.optimizer.step())。
    # 每个批次的损失值被添加到 losses 列表中。
    # 在每个周期结束后，计算并打印平均损失值。
    # 每 10 个周期，将模型权重和优化器状态保存到文件，以便稍后恢复训练。
    # 最后，在所有周期完成后，模型的权重将被保存到文件 f'Unet-epochs{epoch}.pth'。
    # 训练模型的方法
    def train_model(self):
        device = torch.device("cpu")
        loss_list = []
        for epoch in range(self.epochs):
            loss_epoch = []
            dist_positive_epoch = []
            dist_negative_epoch = []
            # 遍历训练数据加载器中的每个批次数据
            for (idx, data) in enumerate(self.train_loader):
                _, inputs = data
                input_var = torch.autograd.Variable(inputs)
                if not (list(input_var.size())[0] == self.batch_size):
                    continue

                inputs_var_batch = input_var.view(self.batch_size * 3, 1, self.image_size, self.image_size)

                # computed output
                # 计算模型输出
                output1 = self.model(inputs_var_batch).to(device)
                output = output1.view(output1.size(0), -1).cpu()
                dist_positive, dist_negative, loss = self.criterion(output)
                if len(dist_positive) == 0 and len(dist_negative) == 0:
                    continue

                # save some metric values
                # 保存一些度量值
                loss_epoch.append(loss.item())
                dist_positive_epoch.append(dist_positive[0].item())
                dist_negative_epoch.append(dist_negative[0].item())

                # compute gradient and do SGD step
                # 计算梯度并执行梯度下降步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_list.append(statistics.mean(loss_epoch))
            mean_dist_positive = statistics.mean(dist_positive_epoch)
            mean_dist_negative = statistics.mean(dist_negative_epoch)

            print(f"Epoch= {epoch:04d}  Loss= {statistics.mean(loss_epoch):0.4f}\
            Mean-Dist-Pos: {mean_dist_positive:0.4f}\
            Mean-Dist-Neg: {mean_dist_negative:0.4f}")
            print("loss", statistics.mean(loss_epoch), epoch)

            # 每5个周期保存一次模型权重和优化器状态
            if epoch % 5 == 0:
                checkpoint = {"epoch": epoch, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                              "loss": loss}

                checkpoint_export_path = os.path.join("my_project/output/HardNet128", f"{self.modelname}.pth")
                torch.save(checkpoint, f'my_project/output/HardNet128/{epoch}.pth"')
                print(f"Checkpoint savec to: {checkpoint_export_path}")