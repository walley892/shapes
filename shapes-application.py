from engine import start_main_loop, Window
from shapes_game_objects import *
from engine.standard_game_objects import *
from engine.standard_components import ObjectMover

if __name__=='__main__':
    g = ScenePatchController()
    c = Camera()
    c.transform.translate(0, 0, -1)
    c.transform.rotate(-3.14/4, 0, 0)
    c.add_component(ObjectMover())
    main_window = Window(game_objects=[g], camera = c)
    main_window.set_size(1280, 720)
    main_window.set_position(0,0)
    sp = SidePanel(g)
    active_patch_window = Window(
        name='Active patches',
        game_objects=[
            sp, 
        ],
        x=1280 + 60,
        y=0,
    )
    active_patch_window.set_size(260, 720)

    bottom_window = Window(
        name='adjust function',
        game_objects=[
            BottomPanel(g, sp),
        ],
        x=0,
        y=720 + 60,
    )
    bottom_window.set_size(1280+260, 260)

    start_main_loop()
