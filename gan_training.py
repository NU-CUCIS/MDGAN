import os, time, itertools, pickle
import numpy as np
import tensorflow as tf
import h5py
import random
from keras import backend
import time
import warnings
warnings.filterwarnings('ignore')

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)
def relu(x):
    return tf.maximum(0., x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, filter_num[0], [4, 4], strides=(2, 2), padding='same', name='conv1')
        relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=isTrain), name='relu1')

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(relu1, filter_num[1], [4, 4], strides=(2, 2), padding='same', name='conv2')
        relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=isTrain), name='relu2')

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(relu2, filter_num[2], [4, 4], strides=(2, 2), padding='same', name='conv3')
        relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=isTrain), name='relu3')

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(relu3, filter_num[3], [4, 4], strides=(2, 2), padding='same', name='conv4')
        relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=isTrain), name='relu4')
        
        # conv5 = tf.layers.conv2d_transpose(relu4, filter_num[4], [4, 4], strides=(2, 2), padding='same')
        # relu5 = relu(tf.layers.batch_normalization(conv5, training=isTrain))
        # output layer

        conv6 = tf.layers.conv2d_transpose(relu4, 1, [4, 4], strides=(2, 2), padding='same', name='conv6')
        o = tf.nn.tanh(conv6, name='output')
        tf.add_to_collection('G_conv1', conv1)
        tf.add_to_collection('G_relu1', relu1)
        tf.add_to_collection('G_conv2', conv2)
        tf.add_to_collection('G_relu2', relu2)
        tf.add_to_collection('G_conv3', conv3)
        tf.add_to_collection('G_relu3', relu3)
        tf.add_to_collection('G_conv4', conv4)
        tf.add_to_collection('G_relu4', relu4)
        tf.add_to_collection('G_conv6', conv6)
        tf.add_to_collection('G_z', o)
        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        # conv0 = tf.layers.conv2d(x, filter_num[4], [4, 4], strides=(2, 2), padding='same')
        # lrelu0 = lrelu(conv0, 0.2)

        conv1 = tf.layers.conv2d(x, filter_num[3], [4, 4], strides=(2, 2), padding='same', name='conv1')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain, name='bn1'), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, filter_num[2], [4, 4], strides=(2, 2), padding='same', name='conv2')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain, name='bn2'), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, filter_num[1], [4, 4], strides=(2, 2), padding='same', name='conv3')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain, name='bn3'), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, filter_num[0], [4, 4], strides=(2, 2), padding='same', name='conv4')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain, name='bn4'), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [8, 8], strides=(1, 1), padding='valid', name='conv5')
        o = tf.nn.sigmoid(conv5, name='output')
        tf.add_to_collection('D_conv1', conv1)
        tf.add_to_collection('D_lrelu1', lrelu1)
        tf.add_to_collection('D_conv2', conv2)
        tf.add_to_collection('D_lrelu2', lrelu2)
        tf.add_to_collection('D_conv3', conv3)
        tf.add_to_collection('D_lrelu3', lrelu3)
        tf.add_to_collection('D_conv4', conv4)
        tf.add_to_collection('D_lrelu4', lrelu4)
        tf.add_to_collection('D_conv5', conv5)
        tf.add_to_collection('D_z', o)
        return o, conv5


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
#     features_mean,features_var =tf.nn.moments(features,axes=[0])    
    features_mean = tf.reduce_mean(features,0)
    features = (features-features_mean)/1
    gram = backend.dot(features, backend.transpose(features))  
    return gram

def style_loss(style, combination):
    loss_temp=0.
    channels = 3
    size = height * width
    
    for i in range(batch_size):
        C = gram_matrix(combination[i])
        S = gram_matrix(style[i])
        loss_temp = tf.add(loss_temp,backend.sum(backend.square(S - C))/(4. * (channels ** 2) * (size ** 2))*3e-1)
    
    return loss_temp

def claps_loss(x):
    z_d_gen=backend.batch_flatten(x)          
    nom = tf.matmul(z_d_gen, tf.transpose(z_d_gen, perm=[1, 0]))
    denom = tf.sqrt(tf.reduce_sum(tf.square(z_d_gen), reduction_indices=[1], keep_dims=True))
    pt = tf.square(tf.transpose((nom / denom), (1, 0)) / denom)
    pt = pt - tf.diag(tf.diag_part(pt))
    pulling_term = tf.reduce_sum(pt) / (batch_size * (batch_size - 1))*4e3
    
    return pulling_term

def conv2d(x, W, stride, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    
def max_pool(x, k_size, stride, padding="SAME"):
    # use avg pooling instead, as described in the paper
    return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], 
            strides=[1, stride, stride, 1], padding=padding)  

def vgg_layers(x):
    ##################  VGG16  ############
    f = h5py.File('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5','r')
    ks = f.keys()
    # get weights and bias of VGG
    vgg16_weights=[]
    vgg16_bias=[]
    for i in range(18):
        if (len(f[ks[i]].values())) != 0:        
            vgg16_weights.append(f[ks[i]].values()[0][:])
            vgg16_bias.append(f[ks[i]].values()[1][:])
        else:
            continue
    del f
    W_conv1 = (tf.constant(vgg16_weights[0]))
    W_conv2 = (tf.constant(vgg16_weights[1]))
    W_conv3 = (tf.constant(vgg16_weights[2]))
    W_conv4 = (tf.constant(vgg16_weights[3]))

    b_conv1 = tf.reshape(tf.constant(vgg16_bias[0]),[-1])
    b_conv2 = tf.reshape(tf.constant(vgg16_bias[1]),[-1])
    b_conv3 = tf.reshape(tf.constant(vgg16_bias[2]),[-1])
    b_conv4 = tf.reshape(tf.constant(vgg16_bias[3]),[-1])

    del vgg16_bias
    del vgg16_weights
    #########  VGG  ################
    # style transfer for generated images
    ######### block 1 ########
    conv_out1 = conv2d(x, W_conv1, stride=1, padding='SAME')
    conv_out1 = tf.nn.bias_add(conv_out1, b_conv1)
    conv_out1 = tf.nn.relu(conv_out1)

    conv_out2 = conv2d(conv_out1, W_conv2, stride=1, padding='SAME')
    conv_out2 = tf.nn.bias_add(conv_out2, b_conv2)
    conv_out2 = tf.nn.relu(conv_out2)
    conv_out2 = max_pool(conv_out2, k_size=2, stride=2, padding="SAME")

    ######### block 2 ########
    conv_out3 = conv2d(conv_out2, W_conv3, stride=1, padding='SAME')
    conv_out3 = tf.nn.bias_add(conv_out3, b_conv3)
    conv_out3 = tf.nn.relu(conv_out3)

    conv_out4 = conv2d(conv_out3, W_conv4, stride=1, padding='SAME')
    conv_out4 = tf.nn.bias_add(conv_out4, b_conv4)
    conv_out4 = tf.nn.relu(conv_out4)
    conv_out4 = max_pool(conv_out4, k_size=2, stride=2, padding="SAME")

    return conv_out1, conv_out2, conv_out3, conv_out4


# training parameters
filter_num = [128, 64, 32, 16] # number of conv filter, five numbers from big to small
feature_vector_dim = (4, 4, 1) # dimension of latent variables
batch_size = 2 # batch size
lr = 0.0005 # learning rate
train_epoch = int(10) # # of training iterations
img_dim = (128,128,1) # dimension of input image
height, width = img_dim[0], img_dim[1]
D_steps = 3 # updating number of discriminator in one training iteration
G_steps = 1 # updating number of generator in one training iteration
cl_loss_weight = 0.03 # weights of model collapse loss
sl_loss_weight = 0.03 # weights of style transfer loss

# load data
with open('./example_data.pkl', 'rb') as f:
    img_collection = pickle.load(f)

# variables : input
x = tf.placeholder(tf.float32, shape=(None,) + img_dim, name='x')
z = tf.placeholder(tf.float32, shape=(None,) + feature_vector_dim, name='z')
isTrain = tf.placeholder(dtype=tf.bool, name='isTrain')
train_set_orig = img_collection.reshape(((len(img_collection),)+ img_dim))
style_array = np.repeat(train_set_orig, 3, axis=-1)
del img_collection

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)


######## style transfer for generated images and real images ########
combination_image_temp=tf.reshape(G_z,[batch_size, height, width, 1])
combination_image = tf.concat([combination_image_temp, combination_image_temp,combination_image_temp], 3)
style_image = tf.placeholder(tf.float32, shape=(batch_size,height,width,3), name='style_image')

conv_out1, conv_out2, conv_out3, conv_out4 = vgg_layers(combination_image)
conv_out1_S, conv_out2_S, conv_out3_S, conv_out4_S = vgg_layers(style_image)
# style loss
sl1 = style_loss(conv_out2_S,conv_out2)
sl2 = style_loss(conv_out4_S,conv_out4)
sl3 = style_loss(conv_out1_S,conv_out1)
sl4 = style_loss(conv_out3_S,conv_out3)
sl_loss = tf.reduce_mean(sl1 + sl2 + sl3 + sl4)
# # claps cost
cl1 = claps_loss(conv_out2)
cl2 = claps_loss(conv_out4)
cl3 = claps_loss(conv_out1)
cl4 = claps_loss(conv_out3)
cl_loss = tf.reduce_mean(cl1 + cl2 + cl3+cl4)


# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
G_loss = G_loss + cl_loss_weight*cl_loss + sl_loss_weight*sl_loss

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-7).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-7).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

D_starting_idx = 0
G_starting_idx = 0
D_num_samples = train_epoch * D_steps * batch_size
G_num_samples = train_epoch * G_steps * batch_size
D_idx_permutation = np.array([])
G_idx_permutation = np.array([])

while D_idx_permutation.shape[0] < D_num_samples:
    D_idx_permutation = np.concatenate((D_idx_permutation, np.random.permutation(train_set_orig.shape[0])), 0)

while G_idx_permutation.shape[0] < G_num_samples:    
    G_idx_permutation = np.concatenate((G_idx_permutation, np.random.permutation(train_set_orig.shape[0])), 0)
D_idx_permutation = D_idx_permutation.astype(int)
G_idx_permutation = G_idx_permutation.astype(int)

for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for D_i in xrange(D_steps):
        trainID = D_idx_permutation[D_starting_idx: D_starting_idx+batch_size]
        x_ = train_set_orig[trainID,:,:,:]
        
        D_starting_idx += batch_size
        z_ = np.random.uniform(0, 1, (batch_size,)+feature_vector_dim)

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)
    for G_i in xrange(G_steps):
        # update generator
        trainID = G_idx_permutation[G_starting_idx: G_starting_idx+batch_size]
        
        G_starting_idx += batch_size

        z_ = np.random.uniform(0, 1, (batch_size,)+feature_vector_dim)
        style_image_input = style_array[trainID, :, :, :]
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, style_image:style_image_input, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

end_time = time.time()
total_ptime = end_time - start_time

print("Training finish!... save training results")
dir_name = './model'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
saver = tf.train.Saver()
saver.save(sess, dir_name+'/model', global_step=train_epoch-1)
sess.close()
