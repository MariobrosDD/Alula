import numpy as np
import pytest
from PIL import Image
from PySide6.QtGui import QIcon, QPixmap

from alula.gui.assets import APP_ICON_ICO, APP_ICON_PNG, LOGO_MARK_PNG

WINDOWS_ICON_SIZES = {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)}


@pytest.mark.parametrize("path", [APP_ICON_PNG, APP_ICON_ICO, LOGO_MARK_PNG])
def test_logo_files_ship_with_the_package(path):
    assert path.is_file()


def test_windows_icon_contains_every_standard_size():
    with Image.open(APP_ICON_ICO) as icon:
        assert WINDOWS_ICON_SIZES <= set(icon.info["sizes"])


def test_app_icon_is_square_with_transparent_corners():
    with Image.open(APP_ICON_PNG) as icon:
        assert icon.width == icon.height
        assert icon.mode == "RGBA"
        assert icon.getpixel((0, 0))[3] == 0
        assert icon.getpixel((icon.width // 2, icon.height - 20))[3] == 255


def white_glyph_margins(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA")).astype(int)
    white = (pixels[:, :, :3].min(axis=2) > 200) & (pixels[:, :, 3] > 200)
    rows, columns = np.nonzero(white)
    height, width = white.shape
    return columns.min(), width - 1 - columns.max(), rows.min(), height - 1 - rows.max()


@pytest.mark.parametrize("path", [APP_ICON_PNG, LOGO_MARK_PNG])
def test_letter_is_centred_in_the_image(path):
    left, right, top, bottom = white_glyph_margins(path)
    assert abs(left - right) <= 2
    assert abs(top - bottom) <= 2


def test_logo_mark_is_white_on_transparent():
    with Image.open(LOGO_MARK_PNG) as mark:
        assert mark.mode == "RGBA"
        pixels = np.asarray(mark)
    opaque = pixels[pixels[:, :, 3] == 255]
    assert len(opaque) > 0
    assert np.all(opaque[:, :3] == 255)
    assert pixels[0, 0, 3] == 0


def test_qt_can_load_the_logo_files(qtbot):
    assert not QIcon(str(APP_ICON_ICO)).isNull()
    assert not QPixmap(str(LOGO_MARK_PNG)).isNull()
