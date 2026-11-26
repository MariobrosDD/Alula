BACKGROUND = "#0a0a0a"
HAIRLINE = "#222222"
FAINT_LINE = "#161616"
PRIMARY_TEXT = "#f2f2f2"
SECONDARY_TEXT = "#8f8f8f"
MUTED_TEXT = "#585858"
WHITE = "#ffffff"
BLACK = "#000000"

DISPLAY_FONT = '"Segoe UI Variable Display Light", "Segoe UI Light", "Helvetica Neue", Arial, sans-serif'
TEXT_LIGHT_FONT = '"Segoe UI Variable Display Semilight", "Segoe UI Semilight", "Segoe UI", sans-serif'
TEXT_FONT = '"Segoe UI Variable Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
MONO_FONT = '"Cascadia Mono", Consolas, "Courier New", monospace'
MONO_FAMILY = "Cascadia Mono"

STYLE_SHEET = f"""
* {{
    border-radius: 0px;
}}
QWidget {{
    color: {PRIMARY_TEXT};
    font-family: {TEXT_FONT};
    font-size: 10pt;
    background: transparent;
}}
QMainWindow, QStackedWidget, #Page, #PageContent {{
    background: {BACKGROUND};
}}
#Sidebar {{
    background: {BACKGROUND};
    border-right: 1px solid {HAIRLINE};
}}
#AppName {{
    font-family: {DISPLAY_FONT};
    font-size: 19pt;
    font-weight: 300;
    letter-spacing: 1px;
    padding: 0px;
}}
#AppTagline {{
    font-family: {MONO_FONT};
    color: {MUTED_TEXT};
    font-size: 8pt;
    padding: 2px 14px 28px 14px;
}}
#NavButton {{
    text-align: left;
    padding: 10px 12px;
    border: none;
    border-left: 2px solid transparent;
    background: transparent;
    color: {SECONDARY_TEXT};
    font-size: 10pt;
}}
#NavButton:hover {{
    color: {PRIMARY_TEXT};
}}
#NavButton:checked {{
    color: {WHITE};
    border-left: 2px solid {WHITE};
}}
#Version {{
    font-family: {MONO_FONT};
    color: {MUTED_TEXT};
    font-size: 8pt;
    padding: 0px 14px;
}}
#PageTitle {{
    font-family: {DISPLAY_FONT};
    font-size: 30pt;
    font-weight: 300;
}}
#Path {{
    font-family: {MONO_FONT};
    color: {MUTED_TEXT};
    font-size: 8.5pt;
}}
#Muted {{
    color: {MUTED_TEXT};
    font-size: 8.5pt;
}}
QPushButton#OutlineButton {{
    background: transparent;
    color: {PRIMARY_TEXT};
    border: 1px solid {SECONDARY_TEXT};
    padding: 9px 18px;
    font-size: 8.5pt;
    letter-spacing: 2px;
}}
QPushButton#OutlineButton:hover {{
    background: {WHITE};
    color: {BLACK};
    border: 1px solid {WHITE};
}}
#FigureStrip {{
    border-top: 1px solid {HAIRLINE};
    border-bottom: 1px solid {HAIRLINE};
}}
#Figure {{
    border-left: 1px solid {HAIRLINE};
}}
#Figure[first="true"] {{
    border-left: none;
}}
#FigureLabel, #SectionTitle {{
    color: {SECONDARY_TEXT};
    font-size: 7.5pt;
    letter-spacing: 2px;
}}
#FigureValue {{
    font-family: {DISPLAY_FONT};
    font-size: 26pt;
    font-weight: 300;
}}
#FigureUnit {{
    font-family: {MONO_FONT};
    color: {SECONDARY_TEXT};
    font-size: 9pt;
}}
#Section {{
    border-top: 1px solid {SECONDARY_TEXT};
}}
#SpecRow {{
    border-bottom: 1px solid {FAINT_LINE};
}}
#RowLabel {{
    font-family: {TEXT_LIGHT_FONT};
    color: {SECONDARY_TEXT};
}}
#RowValue {{
    font-family: {MONO_FONT};
    color: {PRIMARY_TEXT};
    font-size: 9.5pt;
}}
#RowText {{
    color: {PRIMARY_TEXT};
}}
#RowUnit {{
    font-family: {MONO_FONT};
    color: {MUTED_TEXT};
    font-size: 9pt;
}}
QScrollArea {{
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {HAIRLINE};
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{
    background: {SECONDARY_TEXT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    height: 0px;
    background: transparent;
}}
QMenuBar {{
    background: {BACKGROUND};
    color: {SECONDARY_TEXT};
    border-bottom: 1px solid {HAIRLINE};
    padding: 2px 6px;
}}
QMenuBar::item {{
    padding: 5px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    color: {WHITE};
}}
QMenu {{
    background: {BACKGROUND};
    border: 1px solid {HAIRLINE};
    padding: 4px 0px;
}}
QMenu::item {{
    padding: 7px 28px 7px 16px;
    color: {PRIMARY_TEXT};
}}
QMenu::item:selected {{
    background: {WHITE};
    color: {BLACK};
}}
QMenu::separator {{
    height: 1px;
    background: {HAIRLINE};
    margin: 4px 0px;
}}
QStatusBar {{
    background: {BACKGROUND};
    color: {MUTED_TEXT};
    border-top: 1px solid {HAIRLINE};
    font-family: {MONO_FONT};
    font-size: 8pt;
}}
QToolTip {{
    background: {WHITE};
    color: {BLACK};
    border: none;
    padding: 4px 6px;
}}
QMessageBox, QFileDialog {{
    background: {BACKGROUND};
}}
"""
