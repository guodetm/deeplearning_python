import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 读取鸢尾花数据集
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# 使用相同的seed，保证输入特征和标签一一对应
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
t_train = y_data[:-30]
x_test = x_data[-30:]
t_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, dtype=tf.float32)
# t_train = tf.cast(t_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
# tf.cast(t_test, dtype=tf.float32)

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
x_train_batch = tf.data.Dataset.from_tensor_slices((x_train, t_train)).batch(32)
t_test_batch = tf.data.Dataset.from_tensor_slices((x_test, t_test)).batch(32)

# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], mean=0.5, stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([3], mean=0.5, stddev=0.1))
# 设置超参数
lr = 0.1  # 学习率
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和
train_loss_result = []  # 记录下每轮训练的loss值
test_acc=[] # 每轮测试准确率

# 训练部分
for i in range(epoch):
    for step, (x_train, t_train) in enumerate(x_train_batch):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1  # 第一层神经网络运算
            y = tf.nn.softmax(y)  # 经过softmax层将y的输出转换成概率分布
            t_train_label = tf.one_hot(t_train, depth=3)
            loss = tf.reduce_mean(tf.square(t_train_label - y))
            loss_all += loss.numpy()
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])
        # 根据梯度更新参数
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
    # 每个epoch完成，打印loss信息
    print("Epoch {},loss:{}".format(i, loss_all / 4))
    train_loss_result.append(loss_all / 4)
    loss_all = 0
    # 测试部分
    total_correct, total_number = 0, 0  # total_correct为预测正确的样本数,total_number为预测的总样本数
    for x_test, t_test in t_test_batch:
        y_out = tf.matmul(x_test, w1) + b1
        y_out_softmax = tf.nn.softmax(y_out)
        t_out = tf.argmax(y_out_softmax, axis=1)
        pred = tf.cast(t_out, dtype=t_test.dtype)
        correct = tf.cast(tf.equal(pred, t_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:",acc)
    print("--------------------")
