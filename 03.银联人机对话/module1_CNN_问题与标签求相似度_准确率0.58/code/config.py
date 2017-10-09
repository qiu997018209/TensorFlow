#coding:utf-8
'''
Created on 2017年9月26日

@author: qiujiahao

@email:997018209@qq.com

'''
import tensorflow as tf
from collections import namedtuple
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 50, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("sequence_length", 100, "每句话的长度 (default: 0.0)")

# Training parameters
tf.flags.DEFINE_float("threshold", 0.1, "向量间夹角的阈值 (default: 0.05)")
tf.flags.DEFINE_integer("quests_limit", 3, "生成测试数据本标签最低需要的语料要求(default: '4')")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("num_cpu_core", 8, "我的电脑上CPU是8核的")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("file_path", "../data", "File storage path(default: '/root/temp')")


FLAGS = tf.flags.FLAGS

HParams = namedtuple(

  "HParams",
  [
    "embedding_dim",
    "filter_sizes",
    "num_filters",
    "dropout_keep_prob",
    "l2_reg_lambda",
    "sequence_length",
    "threshold",
    "quests_limit",    
    "batch_size",
    "num_epochs",
    "evaluate_every",
    "checkpoint_every",
    "num_cpu_core",
    "allow_soft_placement",
    "log_device_placement",
    "file_path"
  ])

def create_hparams():
    return HParams(
        embedding_dim=FLAGS.embedding_dim,
        filter_sizes=FLAGS.filter_sizes,
        num_filters=FLAGS.num_filters,
        dropout_keep_prob=FLAGS.dropout_keep_prob,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        sequence_length=FLAGS.sequence_length,
        threshold = FLAGS.threshold,
        quests_limit = FLAGS.quests_limit,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        evaluate_every=FLAGS.evaluate_every,
        checkpoint_every=FLAGS.checkpoint_every,
        num_cpu_core = FLAGS.num_cpu_core,
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement, 
        file_path = FLAGS.file_path
    )
    
    
if __name__ == '__main__':
   Flags = create_hparams() 
   for item in Flags._asdict().items():
       print(item[0])