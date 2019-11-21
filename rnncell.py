#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:roger
# datetime:19-11-11 下午5:47
# software: PyCharm
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging


def _get_contact_variable(name, shape, dtype, num_shards):
    """Get a shard variable concatenated into one tensor"""
    sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(sharded_variable) == 1:
        return sharded_variable[0]

    concat_name = name +"/concat"
    concat_full_name = vs.get_variable_scope().name+"/"+concat_name+":0"
    for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
        if value.name == concat_full_name:
            return value
    concat_variable = array_ops.concat(sharded_variable, 0, name=concat_name)
    ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                          concat_variable)
    return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variable with give dtype"""
    if num_shards > shape[0]:
        raise ValueError("Too Many shards: shape=%s, num_shard=%d" % (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards
    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(vs.get_variable(name + "%d" % i, [current_size] + shape[1:], dtype=dtype))
    return shards


class CoupledInputForgetGateLSTMCell(rnn_cell_impl.RNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.
     The default non-peephole implementation is based on:
       http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
     S. Hochreiter and J. Schmidhuber.
     "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
     The peephole implementation is based on:
       https://research.google.com/pubs/archive/43905.pdf
     Hasim Sak, Andrew Senior, and Francoise Beaufays.
     "Long short-term memory recurrent neural network architectures for
      large scale acoustic modeling." INTERSPEECH, 2014.
     The coupling of input and forget gate is based on:
       http://arxiv.org/pdf/1503.04069.pdf
     Greff et al. "LSTM: A Search Space Odyssey"
     The class uses optional peep-hole connections, and an optional projection
     layer.
     """

    def __init__(self, num_units, use_peepholes=False, initializer=None,
                 num_proj=None, proj_clip=None, num_unit_shards=1,
                 num_proj_shards=1, forget_bias=1.0, state_is_tuple=True,
                 activation=math_ops.tanh, reuse=None):
        """Initialize the parameters for an LSTM cell.
        Args:
            num_units:LSTM cell中的单元数量，即隐藏层神经元数量。
            use_peepholes:布尔类型，设置为True则能够使用peephole连接
            cell_clip:可选参数，float类型，如果提供，则在单元输出激活之前，通过该值裁剪单元状态。
            Initializer:可选参数，用于权重和投影矩阵的初始化器。
            num_proj:可选参数，int类型，投影矩阵的输出维数，如果为None，则不执行投影。
            pro_clip:可选参数，float型，如果提供了num_proj>0和proj_clip，则投影值将元素裁剪到[-proj_clip,proj_clip]范围。
            num_unit_shards:弃用。
            num_proj_shards:弃用。
            forget_bias:float类型，偏置增加了忘记门。从CudnnLSTM训练的检查点(checkpoin)恢复时，必须手动设置为0.0。
            state_is_tuple:如果为True，则接受和返回的状态是c_state和m_state的2-tuple；如果为False，则他们沿着列轴连接。后一种即将被弃用。
            activation:内部状态的激活函数。默认为tanh
            reuse:布尔类型，描述是否在现有范围中重用变量。如果不为True，并且现有范围已经具有给定变量，则会引发错误。
        """
        super(CoupledInputForgetGateLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be"
                "deprecated. Use state_is_tuple=True.", self
            )
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._reuse = reuse

        if num_proj:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
                                if state_is_tuple else num_units+num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units)
                                if state_is_tuple else 2*num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        """Run a Step of Lstm
        Args:
            inputs: input tensor, 2D, batch x num_units.
            state: if 'state_is_tuple' is False, this must be a state Tensor,
            '2-D, batch x state_size'. if 'state_is_tuple' is True, this must be a
            tuple of state Tensors, both '2-D' with column size 'c_state' and 'm_state'
         Returns:
        A tuple containing:
            - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
            - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        sigmoid = math_ops.sigmoid
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # Input gate weights
        self.w_xi = tf.get_variable("_w_xi", [input_size.value, self._num_units])
        self.w_hi = tf.get_variable("_w_hi", [self._num_units, self._num_units])
        self.w_ci = tf.get_variable("_w_ci", [self._num_units, self._num_units])

        # Output gate weights
        self.w_xo = tf.get_variable("_w_xo", [input_size.value, self._num_units])
        self.w_ho = tf.get_variable("_w_ho", [self._num_units, self._num_units])
        self.w_co = tf.get_variable("_w_co", [self._num_units, self._num_units])

        # Cell weights
        self.w_xc = tf.get_variable("_w_xc", [input_size.value, self._num_units])
        self.w_hc = tf.get_variable("_w_hc", [self._num_units, self._num_units])

        # Initialize the bias vectors
        self.b_i = tf.get_variable("_b_i", [self._num_units], initializer=init_ops.zeros_initializer())
        self.b_c = tf.get_variable("_b_c", [self._num_units], initializer=init_ops.zeros_initializer())
        self.b_o = tf.get_variable("_b_o", [self._num_units], initializer=init_ops.zeros_initializer())

        i_t = sigmoid(math_ops.matmul(inputs, self.w_xi) +
                      math_ops.matmul(m_prev, self.w_hi) +
                      math_ops.matmul(c_prev, self.w_ci) +
                      self.b_i)

        c_t = ((1 - i_t) * c_prev + i_t * self._activation(math_ops.matmul(inputs, self.w_xc) +
                                                           math_ops.matmul(m_prev, self.w_hc) + self.b_c))

        o_t = sigmoid(math_ops.matmul(inputs, self.w_xo)+
                      math_ops.matmul(m_prev, self.w_ho)+
                      math_ops.matmul(c_t, self.w_co)+
                      self.b_o)

        h_t = o_t * self._activation(c_t)

        new_state = (rnn_cell_impl.LSTMStateTuple(c_t, h_t) if self._state_is_tuple else
                     array_ops.concat([c_t, h_t], 1))
        return h_t, new_state