#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:roger
# datetime:19-11-12 下午5:23
# software: PyCharm

import os
import json
import time
import pickle
import traceback

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options

import tensorflow as tf
from utils import create_model, get_logger
from model import Model
from loader import input_from_line, prepare_dataset,convert
from train import FLAGS, load_config, train

# tornado高并发
import tornado.web
import tornado.gen
import tornado.concurrent
from concurrent.futures import ThreadPoolExecutor

# 定义端口为5000
define("port", default=5000, help="run on the given port", type=int)
# 导入模型
config = load_config(FLAGS.config_file)
logger = get_logger(FLAGS.log_file)
# limit GPU memory
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = False
with open(FLAGS.map_file, "rb") as f:
    tag_to_id, id_to_tag = pickle.load(f)

sess = tf.Session(config=tf_config)
model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)

# 模型训练
class ModelTrainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('begin training...')
        os.system('python3 train.py')
        self.write('end')


# 模型预测的HTTP接口
class ResultHandler(tornado.web.RequestHandler):
    # post函数
    def post(self):
        event = self.get_argument('event')
        lines = self.new_text_split(event)
        inputs = convert(prepare_dataset(lines,FLAGS.max_seq_len,tag_to_id,train=False))
        result = model.evaluate_lines(sess, inputs, id_to_tag)

        self.write(json.dumps(result, ensure_ascii=False))

    @staticmethod
    def new_text_split(text):
        tmp_l = int(FLAGS.max_seq_len) - 2
        pad = 24
        split_index = [[tmp_l * _, tmp_l * (_ + 1) - pad] if _ == 0
                       else [tmp_l * _ - pad, tmp_l * (_ + 1) - pad]
                       for _ in range(int(len(text) / tmp_l) + 1)]
        list_str = []
        for _ in split_index:
            list_str.append(text[_[0]:_[1]])
        return list_str


# 模型预测的HTTP接口
class AsyncResultHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(max_workers=100)

    # get 函数
    @tornado.gen.coroutine
    # post 函数
    def post(self):
        event = self.get_argument('event')
        result = yield self.function(event)
        self.write(json.dumps(result))
        #self.write(str(round(t2 - t1, 4)))

    @tornado.concurrent.run_on_executor
    def function(self, event):
        result = model.evaluate_line(sess, input_from_line(event, FLAGS.max_seq_len, tag_to_id), id_to_tag)
        return result

# 主函数
def main():
    # 开启tornado服务
    tornado.options.parse_command_line()
    # 定义app
    app = tornado.web.Application(
            handlers=[(r'/model_train', ModelTrainHandler),
                      (r'/subj_extract', ResultHandler),
                      (r'/async_subj_extract', AsyncResultHandler)
                     ], #网页路径控制
           )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
