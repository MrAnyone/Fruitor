# import f_trainner
from my_windows.main_runtime import MainRuntime


if __name__ == '__main__':
    runtime = MainRuntime()
    runtime.run()
    # model_params = {
    #     'momentum': 0.9,
    #     'num_epochs': 20
    # }
    # f_trainner.init_module(model_params, rebuild=True)
