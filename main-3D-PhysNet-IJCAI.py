# Created on Tue Oct 17 01:24:46 2017
#
# @author: Zhihua Wang

import os
import shutil
import tensorflow as tf
import scipy.io
import tools
import time
import numpy as np
import copy
import argparse


from datetime import datetime

parser = argparse.ArgumentParser(description='Parse Arguments.')
parser.add_argument('--datasize',type=int,default=400, help='size of dataset')
parser.add_argument('--partition',type=float,default=0.7, help='part of training data')
args = parser.parse_args()



data_size = args.datasize
partition = args.partition


# Parameter Sweep 1: gan Generator losses versus autoencoder loss
para_w = 5
# Parameter Sweep 2: numbers of latent variables in autoencoder
para_f = 5000
# Parameter Sweep 3: gradient penalty
para_p = 10
# data_size = 1000
# partition = 0.7

resolution = 64
batch_size = 8
# lr_down = [0.0005,0.0001,0.00005]
lr_down = [0.0005,0.0001,0.00005,0.00001]
lr_i = 2

GPU0 = '0'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(tf.__version__)
mode = "training"
clear_prior_results = True

result_dir = './Result'

encoding_bits = np.array([1, 1, 1, 1])
train_batches_per_epoch = int(np.floor(partition * data_size/ batch_size))
val_batches_per_epoch = int(np.floor((1-partition) *  data_size/ batch_size))


save_model_dir = os.path.join(result_dir, 'model/')
train_summary_dir = os.path.join(result_dir, 'summary_train')
test_results_dir = os.path.join(result_dir, 'test_results')
test_summary_dir = os.path.join(result_dir, 'summary_test')


if clear_prior_results:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)

        # raise ValueError("Result Folder Already Exists '%s'." % (result_dir))

    os.makedirs(test_results_dir)
    print "Folder '%s'' created!\n" % (test_results_dir)
    os.makedirs(save_model_dir)
    print "Folder '%s'' created!\n" % (save_model_dir)
    os.makedirs(train_summary_dir)
    print "Folder '%s'' created!\n" % (train_summary_dir)
    os.makedirs(test_summary_dir)
    print "Folder '%s'' created!\n" % (test_summary_dir)

def metric_IoU(batch_voxel_occup_pred, batch_voxel_occup_true):
    with tf.name_scope('IOU'):
        batch_voxel_occup_pred_ = copy.deepcopy(batch_voxel_occup_pred)

        batch_voxel_occup_pred_[batch_voxel_occup_pred_ >= 0.5] = 1
        batch_voxel_occup_pred_[batch_voxel_occup_pred_ < 0.5] = 0

        I = batch_voxel_occup_pred_ * batch_voxel_occup_true
        U = batch_voxel_occup_pred_ + batch_voxel_occup_true
        U[U < 1] = 0
        U[U >= 1] = 1

        iou = np.sum(I) * 1.0 / np.sum(U) * 1.0
    return iou


def ae_u(X, C_G):
    with tf.device('/gpu:' + GPU0):
        with tf.name_scope('Encoder'):
            X = tf.reshape(X, [batch_size, resolution, resolution, resolution, 1])
            ##### encode
            c_e = [1, 64, 128, 256, 512]
            s_e = [0, 1, 1, 1, 1]
            layers_e = []
            layers_e.append(X)
            for i in range(1, 5, 1):
                with tf.name_scope('En_Layer' + str(i)):
                    layer = tools.Ops.conv3d(layers_e[-1], k=4, out_c=c_e[i], str=s_e[i],
                                             name='e' + str(i))
                    layer = tools.Ops.maxpool3d(tools.Ops.xxlu(layer, name='lrelu'), k=2, s=2,
                                                pad='SAME')
                    layers_e.append(layer)
            bat, d1, d2, d3, cc = [int(d) for d in layers_e[-1].get_shape()]
            lfc = tf.reshape(layers_e[-1], [bat, -1])

        with tf.name_scope('Ful_Layer1'):
            lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=5000, name='fc1'), name='relu')

        with tf.name_scope('Mean_and_std'):
            z_mean = tools.Ops.fc(lfc, out_d=800, name='z_mean')
            z_std = tools.Ops.fc(lfc, out_d=800, name='z_std')

        with tf.name_scope('Gaussian'):
            # Sampler: Normal (gaussian) random distribution
            eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                                   name='epsilon')
            lfc = z_mean + tf.exp(z_std / 2) * eps

            lfc = tf.concat([lfc, C_G], axis=1)

        with tf.name_scope('Ful_Layer2'):
            lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=d1 * d2 * d3 * cc, name='fc2'),
                                 name='relu')
            lfc = tf.reshape(lfc, [bat, d1, d2, d3, cc])

        with tf.name_scope('Decoder'):
            c_d = [0, 256, 128, 64, 1]
            s_d = [0, 2, 2, 2, 2, 2]
            layers_d = []
            layers_d.append(lfc)
            for j in range(1, 5, 1):
                with tf.name_scope('De_Layer' + str(j)):
                    u_net = True
                    if u_net:
                        layer = tf.concat([layers_d[-1], layers_e[-j]], axis=4)
                        layer = tools.Ops.deconv3d(layer, k=4, out_c=c_d[j], str=s_d[j],
                                                   name='d' + str(len(layers_d)))
                    else:
                        layer = tools.Ops.deconv3d(layers_d[-1], k=4, out_c=c_d[j], str=s_d[j],
                                                   name='d' + str(len(layers_d)))

                    if j != 4:
                        layer = tools.Ops.xxlu(layer, name='relu')
                    layers_d.append(layer)

            vox_no_sig = layers_d[-1]
            vox_sig = tf.sigmoid(layers_d[-1])
            vox_sig_modified = tf.maximum(vox_sig, 0.01)
    return vox_sig, vox_sig_modified, vox_no_sig, z_mean, z_std

def dis( X, Y, C_D):
    with tf.device('/gpu:'+ GPU0):
        X = tf.reshape(X,[batch_size,resolution,resolution,resolution,1])
        Y = tf.reshape(Y,[batch_size,resolution,resolution,resolution,1])
        layer = tf.concat([X,Y],axis=4)
        c_d = [1,64,128,256,512]
        s_d = [0,2,2,2,2]
        layers_d =[]
        layers_d.append(layer)
        for i in range(1,5,1):
            with tf.name_scope('Dis_Layer' + str(i)):
                layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d'+str(i))
                if i!=4:
                    layer = tools.Ops.xxlu(layer, name='lrelu')
                if i == 1:
                    layer = tf.concat([layer, C_D], axis = 4)
            layers_d.append(layer)
        y = tf.reshape(layers_d[-1],[batch_size,-1])
    return tf.nn.sigmoid(y)



def read_tfrecord(filenames):
    dataset = tf.data.TFRecordDataset(filenames,compression_type="ZLIB")


    def parser(record):
        keys_to_features = {
                    'label_name': tf.FixedLenFeature([], tf.string),
                    'shapel': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string),
                    'labelOrigin': tf.FixedLenFeature([], tf.string),
                    'shape1': tf.FixedLenFeature([], tf.string),
                    'pcl1': tf.FixedLenFeature([], tf.string),
                    'shape2': tf.FixedLenFeature([], tf.string),
                    'pcl2': tf.FixedLenFeature([], tf.string),}
        parsed = tf.parse_single_example(record, keys_to_features)
        # pcl was saved as uint8, so we have to decode as uint8.
        label_name = parsed['label_name']
        pcl1 = tf.decode_raw(parsed['pcl1'], tf.uint8)
        shape1 = tf.decode_raw(parsed['shape1'], tf.int32)
        pcl2 = tf.decode_raw(parsed['pcl2'], tf.uint8)
        shape2 = tf.decode_raw(parsed['shape2'], tf.int32)
        label = tf.decode_raw(parsed['label'], tf.float32)
        labelOrigin = tf.decode_raw(parsed['labelOrigin'], tf.float32)
        shapel = tf.decode_raw(parsed['shapel'], tf.int32)

        # the pcl tensor is flattened out, so we have to reconstruct the shape
        reshape_pcl1 = tf.reshape(pcl1, shape1)
        reshape_pcl2 = tf.reshape(pcl2, shape2)
        reshape_label = tf.reshape(label, shapel)
        reshape_labelOrigin = tf.reshape(labelOrigin, shapel)

        # return reshape_pcl1, reshape_pcl2, reshape_label
        return reshape_pcl1, reshape_pcl2, reshape_label, reshape_labelOrigin, label_name

    dataset = dataset.map(parser, num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=16)
    # dataset = dataset.batch(2)
    dataset = dataset.padded_batch(8, padded_shapes=([None, 3],[None, 3],[4],[4],[]))

    dataset = dataset.repeat(2)

    return dataset


def process_condition_real(src):
    encoding_bits_total = np.sum(encoding_bits)
    dst_dis = np.zeros((batch_size, 32, 32, 32, 1))
    dst_gen = np.zeros((batch_size, encoding_bits_total))
    for batch_i in range(len(src)):
        condition = src[batch_i]
        for i in np.arange(len(encoding_bits)):
            dst_dis[ batch_i, i , :, :,0] = condition[i]
        for i in np.arange(len(encoding_bits)):
            dst_gen[ batch_i, i ] = condition[i]
    return dst_gen, dst_dis

def process_voxel(src):
    if len(src)<=0:
        print "Loading point sets error: file empty ",
        exit()
    batch_i = 0
    dst = np.zeros((batch_size, resolution, resolution, resolution, 1))
    for batch in src:
        for i in batch:
            dst[int(batch_i), int(i[0]), int(i[1]), int(i[2]), 0] = 1 # occupied
        batch_i += 1
    return dst


def main():
    with tf.device('/cpu:0'):
        with tf.name_scope('VoxelDataGenerator'):
            tr_dataset = read_tfrecord(['train.tfrecord'])
            val_dataset = read_tfrecord(['test.tfrecord'])
            tr_iterator = tr_dataset.make_initializable_iterator()
            val_iterator = val_dataset.make_initializable_iterator()
            tr_next_element = tr_iterator.get_next()
            val_next_element = val_iterator.get_next()
    with tf.device('/gpu:'+GPU0):
        with tf.variable_scope('placeholder_initialization'):
            X = tf.placeholder(shape=[batch_size, resolution, resolution, resolution, 1],
                               dtype=tf.float32, name='x-input')
            Y = tf.placeholder(shape=[batch_size, resolution, resolution, resolution, 1],
                               dtype=tf.float32, name='y-input')
            C_G = tf.placeholder(shape=[batch_size, np.sum(encoding_bits)],
                                 dtype=tf.float32, name='generator-conditions')
            C_D = tf.placeholder(shape=[batch_size, 32, 32, 32, 1],
                                 dtype=tf.float32, name='discriminator-conditions')
            lr = tf.placeholder(tf.float32)
        
        with tf.variable_scope('autoencoder'):
            Y_pred, Y_pred_modi, Y_pred_nosig, z_mean, z_std = ae_u(X, C_G)
        with tf.variable_scope('discriminator'):
            XY_real_pair = dis(X, Y, C_D)
        with tf.variable_scope('discriminator',reuse=True):
            XY_fake_pair = dis(X, Y_pred, C_D)
        


    with tf.name_scope('train'):

        ################################ ae loss
        Y_ = tf.reshape(Y, shape=[batch_size, -1])
        Y_pred_modi_ = tf.reshape(Y_pred_modi, shape=[batch_size, -1])
        w = 0.85
        with tf.name_scope('reconstruction_loss'):
            # Reconstruction loss
            encode_decode_loss = -tf.reduce_mean(w * Y_ * tf.log(Y_pred_modi_ + 1e-8),
                                                 reduction_indices=[1]) - \
                                 tf.reduce_mean(
                                     (1 - w) * (1 - Y_) * tf.log(1 - Y_pred_modi_ + 1e-8),
                                     reduction_indices=[1])
        with tf.name_scope('kl_Divergence_loss'):
            # KL Divergence loss
            kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

        with tf.name_scope('vae_loss_overall'):
            ae_loss = tf.reduce_mean(encode_decode_loss + kl_div_loss)
            sum_ae_loss = tf.summary.scalar('ae_loss', ae_loss, collections=['loss_summary'])

        ################################ wgan loss
        with tf.name_scope('gan_g_loss'):
            gan_g_loss = -tf.reduce_mean(XY_fake_pair)
            gan_d_loss = tf.reduce_mean(XY_fake_pair) - tf.reduce_mean(XY_real_pair)
            sum_gan_g_loss = tf.summary.scalar('gan_g_loss', gan_g_loss, collections=['loss_summary'])
            sum_gan_d_loss = tf.summary.scalar('gan_d_loss', gan_d_loss, collections=['loss_summary'])

        #Y_pred_ = tf.reshape(Y_pred_modi,shape=[batch_size,-1])
        with tf.name_scope('Y_pred_'):
            # Y_pred_ = tf.reshape(Y,shape=[batch_size,-1])
            Y_pred_ = tf.reshape(Y_pred,shape=[batch_size,-1])

        with tf.name_scope('differences_'):
            differences_ = Y_pred_ -Y_
        with tf.name_scope('interpolates'):
            alpha = tf.random_uniform(shape=[batch_size, resolution ** 3], minval=0.0,
                                      maxval=1.0)
            interpolates = Y_ + alpha*differences_
        with tf.variable_scope('discriminator',reuse=True):
            XY_fake_intep = dis(X, interpolates, C_D)
        with tf.name_scope('gradients'):
            gradients = tf.gradients(XY_fake_intep,[interpolates])[0]
        with tf.name_scope('slopes'):
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
        with tf.name_scope('gradient_penalty'):
            gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
        with tf.name_scope('gan_d_loss'):
            gan_d_loss +=para_p*gradient_penalty

        #################################  ae + gan loss
        with tf.name_scope('ae_and_gan_loss'):
            gan_g_w = para_w
            ae_w = 100-gan_g_w
            ae_gan_g_loss = ae_w * ae_loss + gan_g_w * gan_g_loss


    with tf.name_scope('train_variables'):
        ae_var = [var for var in tf.trainable_variables() if var.name.startswith('autoencoder')]
        dis_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        ae_g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,
                                            epsilon=1e-8).minimize(ae_gan_g_loss, var_list=ae_var)
        dis_optim = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,
                                           epsilon=1e-8).minimize(gan_d_loss,var_list=dis_var)


    print tools.Ops.variable_count()
    merged_summary = tf.summary.merge_all('loss_summary')


    saver = tf.train.Saver(max_to_keep=5)
    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True)

    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()
    checkpoint = tf.train.get_checkpoint_state(save_model_dir)

    print 'checkpoint:', checkpoint

    with tf.Session(config=config) as sess:
        sum_writer_train = tf.summary.FileWriter(train_summary_dir, sess.graph)
        sum_write_test = tf.summary.FileWriter(test_summary_dir)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            load_flag = True
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Can not find existing models"
            load_flag = False
            sess.run(tf.global_variables_initializer())


        max_epoch = 2000
        start_epoch = 1
        print "train_batches_per_epoch:", train_batches_per_epoch


        for epoch in range(start_epoch, max_epoch):

            if epoch == start_epoch + 1:
                start = time.time()

            sess.run(tr_iterator.initializer)



            for i in range(train_batches_per_epoch):
                local_start = time.time()

                #### training
                X_batch, Y_batch, label_batch, reshape_labelOrigin, check_label_name = sess.run(tr_next_element)
                X_batch = process_voxel(X_batch)
                Y_batch = process_voxel(Y_batch)
                X_condition_gen, X_condition_dis = process_condition_real(label_batch)
                
                sess.run([dis_optim],
                         feed_dict={X: X_batch, Y: Y_batch, lr: lr_down[lr_i],
                                    C_G: X_condition_gen, C_D:X_condition_dis},
                         options=run_options,
                         run_metadata=run_metadata)


                sess.run([ae_g_optim],
                         feed_dict={X: X_batch, Y: Y_batch, lr: lr_down[lr_i],
                                    C_G: X_condition_gen, C_D:X_condition_dis},
                         options=run_options,
                         run_metadata=run_metadata)


                Y_pred_c, gan_d_loss_c, ae_loss_c, gan_g_loss_c, summary_sess= sess.run([Y_pred,
                                                                                    gan_d_loss, ae_loss, gan_g_loss,
                                                         merged_summary],
                         feed_dict={X: X_batch, Y: Y_batch,
                                    C_G: X_condition_gen, C_D:X_condition_dis},
                         options=run_options,
                         run_metadata=run_metadata)

                # calculate interations
                iteration = (epoch - start_epoch) * train_batches_per_epoch + (i + 1)

                # Power Concern
                sum_writer_train.add_summary(summary_sess, iteration)



              # calculate IOU
                IOU = metric_IoU(Y_pred_c, Y_batch)

                summary = tf.Summary()
                summary.value.add(tag='train/IOU', simple_value=float(IOU))
                summary.value.add(tag='time/iteration', simple_value=float(iteration))
                sum_writer_train.add_summary(summary, iteration)
                print "epoch:", epoch, " i:",  str(i).zfill(2), " train ae loss: %.6f" % ae_loss_c,\
                    " gan g loss: %.8f"%gan_g_loss_c,\
                    " gan d loss: %.6f"%gan_d_loss_c,\
                    " IOU: %.2f"%IOU,

                
                if epoch == start_epoch:
                    print "   remaining: skipped for first epoch"
                else:
                    # calculate rate
                    rate = (iteration - start_epoch*train_batches_per_epoch ) / (time.time() - start)
                    # calculate duration
                    duration = time.time() - local_start
                    # calculate remaining time (minutes)
                    remaining = (max_epoch - epoch)*train_batches_per_epoch / rate / 60

                    print "     ",\
                        "remaining: %d hours %02d minutes" % divmod(remaining, 60), \
                        "   duration: %d minutes  %02d seconds" % divmod(duration, 60), \
                        "   %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S')


                    summary.value.add(tag='time/rate', simple_value=float(rate))
                    summary.value.add(tag='time/remaining', simple_value=float(remaining))
                    summary.value.add(tag='time/duration', simple_value=float(duration))
                    sum_writer_train.add_summary(summary, iteration)


            #### testing
            if epoch % 200 == 0:
                sess.run(val_iterator.initializer)
                for i in range(val_batches_per_epoch):
                    X_batch, Y_batch, label_batch, reshape_labelOrigin, check_label_name = sess.run(val_next_element)
                    X_batch = process_voxel(X_batch)
                    Y_batch = process_voxel(Y_batch)
                    X_condition_gen, X_condition_dis = process_condition_real(label_batch)


                    ae_loss_t,gan_g_loss_t,gan_d_loss_t, Y_test_pred, Y_test_pred_nosig= \
                        sess.run([ae_loss, gan_g_loss,gan_d_loss, Y_pred,Y_pred_nosig],
                                 feed_dict={X: X_batch, Y: Y_batch,
                                            C_G: X_condition_gen,
                                            C_D:X_condition_dis})

                    IOU = metric_IoU(Y_test_pred, Y_batch)

                    to_save = {'X_test': X_batch, 'Y_test_pred': Y_test_pred,'Y_test_true': Y_batch}
                    scipy.io.savemat(
                        test_results_dir + '/X_Y_pred_' + str(epoch).zfill(2) + '_' + str(
                            i).zfill(4) + '.mat', to_save, do_compression=True)

                    print "epoch:", epoch, " i:", i, " test ae loss: %.8f" % ae_loss_t,\
                        " gan g loss: %.8f"%gan_g_loss_c,\
                        " gan d loss: %.8f"%gan_d_loss_c,\
                        " IOU: %.4f"%IOU
                    print "evaluation result saved"


            #### model saving
            if epoch % 50 == 0:

                saver.save(sess, save_path=save_model_dir + "model",
                           global_step=epoch)
                print "Model saved at epoch:", epoch, " saved models are:", saver.last_checkpoints


if __name__ == "__main__":

    main()
