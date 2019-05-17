from .fruitor_model_trainer import FTrainer
import sys
import os

fruitor_model = None


def init_module(params, rebuild=False, model_name='fruitor_model_linear.pt'):
    global fruitor_model
    files = os.listdir('./f_trainner')
    found = False
    fruitor_model = FTrainer()

    for item in files:
        if item == model_name:
            found = True
            print('model found => {}'.format(model_name))
            break
    if not found or rebuild:
        fruitor_model.train(params)
    else:
        fruitor_model.load(path='./f_trainner/' + model_name)
