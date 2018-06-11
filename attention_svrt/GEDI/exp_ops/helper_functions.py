import re, os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from glob import glob
import seaborn as sns
import pandas as pd

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def fine_tune_prepare_layers(tf_vars,finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        if re.split('/',v.name)[0] in finetune_vars:
            ft_vars.append(v)
        else:
            other_vars.append(v)
    return other_vars, ft_vars

def ft_optimized(cost,var_list_1,var_list_2,optimizer,lr_1,lr_2): #applies different learning rates to specified layers
    opt1 = optimizer(lr_1)
    opt2 = optimizer(lr_2)
    grads = tf.gradients(cost, var_list_1 + var_list_2)
    grads1 = grads[:len(var_list_1)]
    grads2 = grads[len(var_list_1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list_1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list_2))
    return tf.group(train_op1, train_op2)

def ft_non_optimized(cost,other_opt_vars,ft_opt_vars,optimizer,lr_1,lr_2):
    op1 = tf.train.AdamOptimizer(lr_1).minimize(cost, var_list=other_opt_vars)
    op2 = tf.train.AdamOptimizer(lr_2).minimize(cost, var_list=ft_opt_vars)
    return tf.group(op1,op2) #ft_optimize is more efficient. siwtch to this once things work

def class_accuracy(pred,targets):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pred,1), tf.cast(targets,dtype=tf.int64)))) #assuming targets is an index

def softmax_cost(activations,labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(activations, labels))

def find_ckpts(config,dirs=None):
    if dirs is None:
        dirs = sorted(glob(config.train_checkpoint + config.which_dataset + '*'),reverse=True)[0] #only the newest model run
    ckpts = glob(dirs + '/*.ckpt*')
    ckpts = [ck for ck in ckpts if '.meta' in ck] #Do not include meta files
    ckpts=[ck[:-5] for ck in ckpts]
    ckpt_names = sorted([int(re.split('-',ck)[-1]) for ck in ckpts])
 
    ckpt_names=[str(i) for i in ckpt_names]
    ckpts=[dirs+'/model_'+i+'.ckpt-'+i for i in ckpt_names]
  
    return ckpts, ckpt_names

def acc_index(accs,ckpt_names):
    out = []
    for idx,acc in enumerate(accs):
        out = np.append(out,np.repeat(idx,len(acc)))
    return out

def plot_accuracies(accs,ckpt_names,output_file):
    df = pd.DataFrame(np.vstack((np.hstack((accs[:])) * 100,acc_index(accs,ckpt_names))).transpose(),columns=['Validation image batch accuracies','Training iteration'])
    #ax = sns.stripplot(x='Training iteration', y='Validation image batch accuracies',data=df, jitter=True)
    sns.set(style="whitegrid")
    ax = sns.boxplot(x='Training iteration', y='Validation image batch accuracies', data=df, whis=np.inf)
    ax = sns.swarmplot(x='Training iteration', y='Validation image batch accuracies', data=df)
    ax.set_ylabel('Validation image batch classification accuracy')
    ax.set_xlabel('Model training iteration')
    ax.set_xticklabels(ckpt_names)
    ax.set(ylim=(50,105))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:2.0f}%'.format(x) for x in vals])
    plt.savefig(output_file)


