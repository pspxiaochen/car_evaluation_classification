import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_processing

data = data_processing.load_data(Download = False)
new_data = data_processing.oneHot(data)

new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data) # 打乱数据的顺序

sub = int(0.7 * len(new_data))
train_data = new_data[:sub]
test_data = new_data[sub:]
batch_index = np.random.randint(len(train_data),size=32)
#########开始构建神经网络
tf_input = tf.placeholder(tf.float32,[None,25],name='input')
tf_x = tf_input[:,:21]
tf_y = tf_input[:,21:]

l1 = tf.layers.dense(inputs=tf_x,units=128,activation=tf.nn.relu,name='l1')
l2 = tf.layers.dense(inputs=l1,units=128,activation=tf.nn.relu,name='l2')
out = tf.layers.dense(inputs=l2,units=4,name='out')
prediction = tf.nn.softmax(out,name='pred')

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=out)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(prediction,axis=1))[1]
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
accuracies,steps = [], []
for i in range(4000):
    batch_index = np.random.randint(len(train_data),size=32)
    sess.run(train_op,feed_dict={tf_input:train_data[batch_index]})

    if i % 50 == 0:
        acc_,pred_,loss_ = sess.run([accuracy,prediction,loss],feed_dict={tf_input:test_data})
        accuracies.append(acc_)
        steps.append(i)
        print('Step: %i' % i,'|Accurate:%.2f' % acc_,'|Loss:%2f' % loss_)
    ax1.cla()
    for c in range(4):
        bp = ax1.bar(left =c + 0.1, height=sum((np.argmax(pred_, axis=1) == c)), width=0.2, color='red')
        bt = ax1.bar(left =c - 0.1, height=sum((np.argmax(test_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
    ax1.set_xticks(range(4), ["accepted", "good", "unaccepted", "very good"])
    ax1.legend(handles=[bp, bt], labels=["prediction", "target"])
    ax1.set_ylim((0, 400))
    ax2.cla()
    ax2.plot(steps, accuracies, label="accuracy")
    ax2.set_ylim(ymax=1)
    ax2.set_ylabel("accuracy")
    plt.pause(0.01)
plt.ioff()
plt.show()