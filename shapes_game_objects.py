from engine.game_object import GameObject
from engine.rendering.standard_renderers import ColoredPointCloudRenderer
from engine import Window
from shapes import *
from engine.standard_game_objects import ButtonPanel, Button, TextInputField, ButtonTable, ButtonLayout

class GamePatch(GameObject):
    def __init__(self, patch):
        super().__init__()
        self.patch = patch
        self.add_renderer(ColoredPointCloudRenderer())
        self.renderer = self.renderers[0]

    @property
    def patch(self):
        return self._patch

    @patch.setter
    def patch(self, p):
        self._patch = p
        self.transform.reset()
        self.transform.translate(-(p.bounds[0]+(abs(p.bounds[0]-p.bounds[1])/2)),-(p.bounds[2]-(abs(p.bounds[2]-p.bounds[3])/2)), -max(abs(p.bounds[0] - p.bounds[1]), abs(p.bounds[2]- p.bounds[3])))

    def update_mesh(self):
        self.renderer.vertex_buffer = self._patch.get_verts()
        self.renderer.color_buffer = self._patch.get_colors()
        self.renderer.n_points = len(self.renderer.vertex_buffer)

    def update(self):
        self._patch.compute()
        self.update_mesh()

class PatchCreationController(GameObject):
    def __init__(self, patch_controller):
        super().__init__()
        self.patch_controller = patch_controller
        '''
        self.text_input = TextInputField('lmao', 1, 1, lambda x: x)
        self.text_input.transform.translate(0,0,-1.3)
        self.text_input.set_parent(self)
        '''
        self.button_panel = ButtonPanel(1, 0.4, [], [],layout= ButtonLayout.VERTICAL)
        self.button_panel.transform.translate(0,0,-1.3)
        self.button_panel.set_parent(self)
        self.button_panel.add_button(Button(lambda x: x, 1, 0.5, text='Define patch function PHI', color=(0, 0, 0)))
        self.button_panel.add_button(Button(lambda x: x, 1, 1, text='PHI_x(u, v, t) == ', color=(0, 0, 1)))
        self.phi_x_input = TextInputField('u', 1, 1, lambda x: x)
        self.button_panel.add_button(self.phi_x_input)
        self.button_panel.add_button(Button(lambda x: x, 1, 1, text='PHI_y(u, v, t) == ', color=(0, 0, 1)))
        self.phi_y_input = TextInputField('v', 1, 1, lambda x: x)
        self.button_panel.add_button(self.phi_y_input)
        self.button_panel.add_button(Button(lambda x: x, 1, 1, text='PHI_z(u, v, t) == ', color=(0, 0, 1)))
        self.phi_z_input = TextInputField('sin(t)', 1, 1, lambda x: x)
        self.button_panel.add_button(self.phi_z_input)
        self.button_panel.add_button(Button(self.create_new_patch_cb, 1, 1, text='Add patch', color=(0, 0, 1)))

    def create_new_patch_cb(self, button):
        patch_string = ','.join([
            self.phi_x_input.text,
            self.phi_y_input.text,
            self.phi_z_input.text,
        ])
        self.patch_controller.add_patch(
            GamePatch(
                StringPatch(
                    patch_string,
                    updates = 't' in patch_string,
                )
            )
        )
        self.window.destroy()

class ActivePatchPanelController(GameObject):
    def __init__(self, patch_controller):
        super().__init__()
        self.patch_controller = patch_controller
        self.button_panel = ButtonPanel(1, 1, [self.spawn_patch_creation_ui_cb], ['Add New ∈ '], [(0.3, 0.2, 0)])
        self.button_panel.transform.translate(0,0,-1.3)
        self.button_panel.set_parent(self)
        self.new_patch_creation_window = None
        self.item_height = 160

    def spawn_patch_creation_ui_cb(self, button):
        if self.new_patch_creation_window is None or not self.new_patch_creation_window.is_alive:
            self.new_patch_creation_window = Window(
                name='Patch Creation',
                game_objects=[
                    PatchCreationController(self.patch_controller),
                ],
                x=self.window.x + 260,
                y=0,
            )
            self.new_patch_creation_window.set_size(260, 3*160)

    def update_panel(self):
        if self.button_panel.n_buttons  - 1 != self.patch_controller.n_patches:
            self.button_panel.remove_parent()
            del self.button_panel
            self.button_panel = ButtonPanel(1, 1, [], [], layout = ButtonLayout.VERTICAL)
            self.button_panel.set_parent(self)
            self.button_panel.transform.translate(0,0,-1.3)
            for i, patch in enumerate(self.patch_controller.get_patches()):
                color = [0,0,0]
                color[i%3] = 1
                self.button_panel.add_button(
                    Button(lambda x: x,
                        1, 
                        1,
                        text=patch.patch.patch_string,
                        color=color
                    )
                )
            self.button_panel.add_button(
                Button(self.spawn_patch_creation_ui_cb,
                    1,
                    1,
                    text='Add new patch',
                    color=(0.3, 0.2, 0)
                )
            )
            self.resize_window()


    def update(self):
        self.update_panel()

    def resize_window(self):
        window_height = self.item_height*(self.patch_controller.n_patches + 1)
        self.window.set_size(260, window_height)

class SidePanel(GameObject):
    def __init__(self, patch_controller):
        super().__init__()
        self.buttons = [
            [Button(lambda x: x, 1, 1, text='Domain', color=(0,0,0))], 
            [Button(self.square_domain_cb,1, 1, text='Square', color=(0,0.2,0)), Button(self.circle_domain_cb,1, 1 , text='Circle', color=(0.2, 0, 0))], 
            [Button(lambda x: x, 1, 1, text='u ∈', color=(0,0,0)), TextInputField('(0, 1)', 1, 1, lambda x: x, color=(0.1,0,0))],
            [Button(lambda x: x, 1, 1, text='v ∈', color=(0,0,0)), TextInputField('(0, 1)', 1, 1, lambda x: x, color=(0.1,0,0.1))],
            [Button(lambda x: x, 1, 1, text='t ∈', color=(0,0,0)), TextInputField('(0, 1)', 1, 1, lambda x: x, color=(0.1,0,0.1))],
            [Button(self.gaussian_curvature_cb, 1, 1, text='Gaussian Curvature', color=(0.2,0,0))], 
            [Button(self.normal_curvature_cb, 1, 1, text='Mean Curvature', color=(0.2,0,0))], 
            [Button(self.animate_cb, 1, 1, text='Auto Animate', color=(0.2,0,0))], 
            [Button(lambda x: x, 1, 1, text='t=0', color=(0,0.2,0)), Button(self.pause_play_cb, 1, 1, text='||', color=(0, 0.2,0))], 
        ]
        h_weights = [
            [1],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.3, 0.7],
            [0.3, 0.7],
            [1],
            [1],
            [1],
            [0.5, 0.5],
        ]
        self.button_panel = ButtonTable(1, 1, self.buttons, h_weights = h_weights)
        self.button_panel.set_parent(self)
        self.button_panel.transform.translate(0, 0, -1.3)
        self.patch_controller = patch_controller

    def update(self):
        patch = self.patch_controller.get_patch()
        if patch is not None:
            if not patch.is_paused:
                self.buttons[-1][0].update_text('t=' + str(round(patch._time, 2)))

    def pause_play_cb(self, button):
        patch = self.patch_controller.get_patch()
        if patch is not None:
            if not patch.is_paused:
                button.update_color((0.2, 0, 0))
                button.update_text('o')
            else:
                button.update_color((0, 0.2, 0))
                button.update_text('||')

            patch.is_paused = not patch.is_paused
        
            

    def get_bounds_string(self):
        us = self.buttons[2][1].text.strip(' ').strip('(').strip(')').split(',')
        vs = self.buttons[3][1].text.strip(' ').strip('(').strip(')').split(',')
        return '({}, {}, {}, {})'.format(us[0], us[1], vs[0], vs[1])
    def get_time_bounds_string(self):
        return self.buttons[4][1].text
    def update_u_bounds(self, bounds):
        self.buttons[2][1].update_text(bounds)
    def update_v_bounds(self, bounds):
        self.buttons[3][1].update_text(bounds)
    def update_time_bounds(self,bounds):
        self.buttons[4][1].update_text(bounds)

    def clear_selections(self):
        self.buttons[6][0].update_color([0.2,0,0])
        self.buttons[5][0].update_color([0.2,0,0])
        self.buttons[7][0].update_color([0.2,0,0])
        self.buttons[1][0].update_color([0,0.2,0])
        self.buttons[1][1].update_color([0.2,0,0])
    
    def gaussian_curvature_cb(self, button):
        current_curvature = self.patch_controller.get_patch().curvature_type
        if current_curvature == 1:
            self.patch_controller.get_patch().curvature_type = 0
            button.update_color([0.2, 0, 0])
            return
        self.patch_controller.get_patch().curvature_type = 1
        button.update_color([0, 0.2, 0])
        self.buttons[6][0].update_color([0.2, 0, 0])
    
    def normal_curvature_cb(self, button):
        current_curvature = self.patch_controller.get_patch().curvature_type
        if current_curvature == 2:
            self.patch_controller.get_patch().curvature_type = 0
            button.update_color([0.2, 0, 0])
            return
        self.patch_controller.get_patch().curvature_type = 2
        button.update_color([0, 0.2, 0])
        self.buttons[5][0].update_color([0.2, 0, 0])

    def square_domain_cb(self, button):
        self.patch_controller.get_patch().is_square = True
        self.buttons[2][0].update_text('u ∈')
        self.buttons[3][0].update_text('v ∈')
        button.update_color([0, 0.2, 0])
        self.buttons[1][1].update_color([0.2, 0, 0])
    def circle_domain_cb(self, button):
        self.patch_controller.get_patch().is_square = False
        self.buttons[2][0].update_text('r ∈')
        self.buttons[3][0].update_text('θ ∈')
        button.update_color([0, 0.2, 0])
        self.buttons[1][0].update_color([0.2, 0, 0])

    def animate_cb(self, button):
        self.patch_controller.get_patch().animate = not self.patch_controller.get_patch().animate
        if self.patch_controller.get_patch().animate:
            button.update_color([0, 0.2, 0])
        else:
            button.update_color([0.2, 0, 0])

        

class BottomPanel(GameObject):
    def __init__(self, patch_controller, side_panel):
        super().__init__()
        self.buttons = [
            [Button(lambda x: x, 1, 1, text='x(u,v,t)=', color=(0,0,0)), TextInputField('u', 1, 1, lambda x: x, color=(0.1,0,0.1))],
            [Button(lambda x: x, 1, 1, text='y(u,v,t)=', color=(0,0,0)), TextInputField('v', 1, 1, lambda x: x, color=(0.1,0,0.1))],
            [Button(lambda x: x, 1, 1, text='z(u,v,t)=', color=(0,0,0)), TextInputField('0', 1, 1, lambda x: x, color=(0.1,0,0.1))],
            [Button(self.create_new_patch_cb, 1, 1, text='Submit', color=(0.2,0,0)), Button(self.create_new_random_patch_cb, 1, 1, text='Randomize', color=(0.2,0,0))], 
        ]
        h_weights = [
            [0.15, 0.85],
            [0.15, 0.85],
            [0.15, 0.85],
            [0.5, 0.5],
        ]

        self.button_panel = ButtonTable(1, 1, self.buttons, h_weights=h_weights)
        self.button_panel.set_parent(self)
        self.button_panel.transform.translate(0, 0, -1.3)
        self.patch_controller = patch_controller
        self.side_panel = side_panel
    
    def bounds_str_to_tuple(self, bounds_string):
        return tuple(map(float, bounds_string.strip('(').strip(')').split(',')))

    def create_new_patch_cb(self, button):
        patch_string = ','.join([
            self.buttons[0][1].text,
            self.buttons[1][1].text,
            self.buttons[2][1].text,
        ])
        self.patch_controller.update_patch(
            StringPatch(
                patch_string,
                updates = 't' in patch_string,
                bounds_string = self.side_panel.get_bounds_string(), 
                time_bounds=self.bounds_str_to_tuple(self.side_panel.get_time_bounds_string())
            )
        )
        self.side_panel.clear_selections()

    def create_new_random_patch_cb(self, button):
        patch = FourierPatch(
            True, 
            time_bounds=self.bounds_str_to_tuple(self.side_panel.get_time_bounds_string())
        )
        self.patch_controller.update_patch(
            patch
        )
        self.buttons[0][1].update_text(patch.phi_x())
        self.buttons[1][1].update_text(patch.phi_y())
        self.buttons[2][1].update_text(patch.phi_z())
        self.side_panel.clear_selections()
        self.side_panel.update_u_bounds(patch.u_bounds())
        self.side_panel.update_v_bounds(patch.v_bounds())
        self.side_panel.update_time_bounds(str(patch.time_bounds))

class ScenePatchController(GameObject):
    def __init__(self):
        super().__init__()
        self.patches = {}
        pass
    
    def get_patch(self):
        for patch in self.patches.values():
            return patch.patch
    def update_patch(self, patch):
        if len(self.patches) == 0:
            self.add_patch(GamePatch(patch))
        else:
            for patchh in self.patches.values():
                patchh.patch = patch
                return

    def clear_patches(self):
        patches = [patch for patch in self.patches]
        for patch in patches:
            self.remove_patch(self.patches[patch])

    def add_patch(self, game_patch):
        self.clear_patches()
        game_patch.set_parent(self)
        self.patches[str(game_patch)] = game_patch

    def remove_patch(self, game_patch):
        game_patch.remove_parent()
        if str(game_patch) in self.patches:
            del self.patches[str(game_patch)]

    def get_patches(self):
        return list(self.patches.values())

    @property
    def n_patches(self):
        return len(self.patches)

