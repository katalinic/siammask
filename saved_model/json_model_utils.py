import tensorflow as tf
import numpy as np
import json


def set_model_params(sess, model_vars, saved_params):
    assign_ops = []
    for model_var, saved_param in zip(model_vars, saved_params):
        name, value = saved_param
        msg = 'Variable names do not match: {} {}'.format(model_var.name, name)
        assert model_var.name == name, msg
        msg = 'Variable shapes for variable {} do not match: {} {}'.format(
            model_var.name, model_var.shape, value.shape)
        assert model_var.shape == value.shape, msg
        assign_ops.append(model_var.assign(value))
    assign_ops_group = tf.group(*assign_ops)
    sess.run(assign_ops_group)


def load_json(sess, vars_to_load, jsonfile='model.json'):
    with open(jsonfile, 'r') as f:
        saved_params = json.load(f)
    saved_params = [(p[0], np.array(p[1])) for p in saved_params]
    set_model_params(sess, vars_to_load, saved_params)
