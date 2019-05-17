import pygame
import cv2
import f_trainner
import numpy as np
from .text_box import TextBox


class MainRuntime:
    def __init__(self, ):
        pygame.init()
        self.settings = {
            'width': 720,
            'height': 360,
            'full_screen': False,
            'stay_open': True,
            'fps': 60
        }
        model_params = {
            'momentum': 0.9,
            'num_epochs': 20
        }
        f_trainner.init_module(model_params)

    def run(self):
        window = pygame.display.set_mode((self.settings['width'], self.settings['height']), flags=pygame.FULLSCREEN if self.settings['full_screen'] else 0)
        pygame.display.set_caption('FRUITOR')
        cap = cv2.VideoCapture(0)
        clock = pygame.time.Clock()
        text_box = {
            'predict': TextBox(editable=False, text='Waiting...', pos=(520, 50), background_color=(206, 10, 10, 200), text_pos=(10,0)),
            'possibility_1': TextBox(editable=False, text='Waiting...', pos=(520, 90),text_pos=(10,0)),
            'possibility_2': TextBox(editable=False, text='Waiting...', pos=(520, 130), text_pos=(10,0)),
            'possibility_3': TextBox(editable=False, text='Waiting...', pos=(520, 170), text_pos=(10,0)),
            'possibility_4': TextBox(editable=False, text='Waiting...', pos=(520, 210), text_pos=(10,0)),
        }
        counter = 0

        while self.settings['stay_open']:
            window.fill((0, 0, 0))
            clock.tick(self.settings['fps'])
            ret, frame = cap.read()
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(cv2_im)
            frame = pygame.surfarray.make_surface(frame)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.settings['stay_open'] = False
                elif event.type == pygame.KEYDOWN:
                    # cv2.imwrite("./datasets/test_set/frame_{}".format(counter), frame)
                    result = f_trainner.fruitor_model.predict(cv2_im)
                    text_box['predict'].change_text_value(result[0])
                    text_box['possibility_1'].change_text_value(result[1])
                    text_box['possibility_2'].change_text_value(result[2])
                    text_box['possibility_3'].change_text_value(result[3])
                    text_box['possibility_4'].change_text_value(result[4])
            for box in text_box:
                text_box[box].render_part(window)
            frame = pygame.transform.scale(frame, (504, 360))
            window.blit(frame, (0, 0))
            pygame.display.update()
