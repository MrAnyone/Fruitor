import pygame


class TextBox:

    def __init__(self, text='', actif=False, editable=False, entry_hider=None,
                 font_path='./my_windows/font/Jamma 8x16.ttf', box_size=(250, 30),
                 box_background=None, background_color=(150, 150, 150, 200), text_underline=False,
                 size=20, hit_text='', color=(250, 250, 250), pos=(0, 0), text_pos=(0, 0)):
        self.editable = editable
        self.hit_text = hit_text
        self.text_pos = text_pos
        self.pos = pos
        self.font_path = font_path
        self.size = size
        self.box_size = box_size
        self.actif = actif
        self.text = text
        self.on_screen = False
        self.entry_hider = entry_hider
        self.color = color
        self.text_underline = text_underline
        self.background_color = pygame.Color(background_color[0], background_color[1],
                                                 background_color[2], background_color[3])
        self.font = pygame.font.Font(font_path, size)
        if self.text_underline:
            self.font.set_underline(True)
        self.show_cursor = True
        self.box_background = box_background
        if box_background:
            self.box_background = pygame.image.load(box_background).convert_alpha()
            self.box_background = pygame.transform.scale(self.box_background, box_size)
            self.box_surface = self.box_background
        else:
            self.box_surface = pygame.Surface(box_size, pygame.SRCALPHA)
            self.box_surface.fill(self.background_color)
        self.box_rect = self.box_surface.get_rect()
        self.box_rect.move_ip(pos[0], pos[1])

    def change_text_value(self, text=''):
        self.font.render('', False, self.color)
        self.box_surface.fill(self.background_color)
        self.text = text

    def update(self, font=None, box_size=None, size=None, box_background=None, background_color=None):
        font_change = (
            font if font else self.font_path,
            size if size else self.size
        )
        if font_change[0] != self.font_path or font_change[1] != self.size:
            self.font = pygame.font.Font(font_change[0], font_change[1])
            if self.text_underline:
                self.font.set_underline(True)
        if box_size:
            self.box_rect = pygame.Surface(box_size)
        if box_background:
            self.box_surface = pygame.image.load(box_background).convert_alpha()
            self.box_surface = pygame.transform.scale(self.box_surface, box_size)
        if background_color:
            self.box_rect.fill(background_color)
        self.font = pygame.font.Font(font, size)

    def trigger_input(self, input_key):
        if self.actif and self.editable:
            if self.box_background:
                self.box_surface = self.box_background
            else:
                self.box_surface.fill(self.background_color)
            if input_key == 8:
                self.text = self.text[:-1]
            else:
                self.text += chr(input_key)

    def trigger_click(self, pos):
        if self.box_rect.collidepoint(pos):
            if self.text == '':
                if self.box_background:
                    self.box_surface = self.box_background
                else:
                    self.box_surface.fill(self.background_color)
            self.actif = True
            self.show_cursor = True
        elif self.actif:
            self.actif = False

    def render_part(self, surface):
        text_to_render = self.text if self.text != '' else self.hit_text
        if self.actif:
            if self.entry_hider:
                text_to_render = str('').zfill(len(self.text)).replace('0', self.entry_hider)
            else:
                text_to_render = self.text
            if self.show_cursor:
                text_to_render += "|"
        else:
            if self.show_cursor:
                self.show_cursor = False
                if self.box_background:
                    self.box_surface = self.box_background
                else:
                    self.box_surface.fill(self.background_color)
                text_to_render = text_to_render[:-1]
            if self.text == '':
                if self.box_background:
                    self.box_surface = self.box_background
                else:
                    self.box_surface.fill(self.background_color)
                text_to_render = self.hit_text
            elif self.text != '' and self.entry_hider:
                text_to_render = str('').zfill(len(self.text)).replace('0', self.entry_hider)
        self.on_screen = True
        text_surface = self.font.render(text_to_render, False, self.color)
        self.box_surface.blit(text_surface, self.text_pos)
        surface.blit(self.box_surface, self.pos)
