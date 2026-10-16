"""El panel LoRa cuenta e indica actividad de cada sync word por separado."""


def test_lora_sync_word_cards_count_and_blink_independently(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ui_builder import LoRaSyncWordPanel

    app = QApplication.instance() or QApplication([])
    panel = LoRaSyncWordPanel()
    panel.show()

    panel.record(0x12, b'A', True)
    panel.record(0x34, b'B', None)
    panel.record(0x12, b'C', False)

    assert list(panel.entries) == [0x12, 0x34]
    assert panel.entries[0x12]['count'] == 2
    assert panel.entries[0x34]['count'] == 1
    assert panel.entries[0x12]['counter'].text() == 'Paquetes: 2'
    assert panel.scroll_area.height() == 98
    assert '#4be16b' in panel.entries[0x12]['led'].styleSheet()

    QTest.qWait(120)
    panel.record(0x12, b'D', True)
    QTest.qWait(130)
    assert '#4be16b' in panel.entries[0x12]['led'].styleSheet()
    assert '#38413b' in panel.entries[0x34]['led'].styleSheet()

    QTest.qWait(120)
    assert '#38413b' in panel.entries[0x12]['led'].styleSheet()
    assert panel.entries[0x12]['count'] == 3

    panel.reset()
    assert panel.entries == {}
    assert not panel.placeholder.isHidden()
    assert panel.scroll_area.height() == 38
    panel.close()


def test_lora_sync_word_cards_stay_below_config_in_tall_sidebar(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    from ui_builder import LoRaSyncWordPanel

    app = QApplication.instance() or QApplication([])
    sidebar = QWidget()
    layout = QVBoxLayout(sidebar)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    config = QLabel('BW: 500 kHz\nSF: 12')
    config.setFixedHeight(95)
    layout.addWidget(config)
    panel = LoRaSyncWordPanel()
    layout.addWidget(panel)
    sidebar.resize(300, 1047)
    sidebar.show()

    panel.record(0x34, b'example', True)
    app.processEvents()

    assert panel.y() - (config.y() + config.height()) <= 10
    assert panel.scroll_area.y() <= 35
    assert panel.height() <= 90
    sidebar.close()


def test_lora_sync_word_history_opens_from_card_and_updates_live(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtWidgets import QApplication
    from ui_builder import LoRaSyncWordPanel

    app = QApplication.instance() or QApplication([])
    panel = LoRaSyncWordPanel()
    panel.show()
    panel.record(0x12, b'\xffA\x00', True)
    panel.record(0x12, b'hello', False)
    panel.record(0x34, b'other', None)

    panel.entries[0x12]['history_button'].click()
    dialog = panel.entries[0x12]['dialog']
    app.processEvents()
    assert dialog.isVisible()
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(1, 3).text() == 'Inválido'
    dialog.table.selectRow(0)
    assert dialog.hex_payload.toPlainText() == 'FF 41 00'
    assert dialog.text_payload.toPlainText() == 'A'
    dialog.full_text_checkbox.setChecked(True)
    assert dialog.text_payload.toPlainText() == r'\xFFA\x00'

    panel.record(0x12, b'new', True)
    assert dialog.table.rowCount() == 3
    assert panel.entries[0x12]['count'] == 3
    panel.entries[0x34]['history_button'].click()
    other_dialog = panel.entries[0x34]['dialog']
    assert other_dialog.table.rowCount() == 1
    assert other_dialog.table.item(0, 3).text() == 'No presente'

    panel.close_histories()
    assert not dialog.isVisible()
    assert not other_dialog.isVisible()
    panel.entries[0x12]['history_button'].click()
    assert dialog.isVisible()

    panel.reset()
    app.processEvents()
    assert panel.entries == {}
    assert not dialog.isVisible()
    assert not other_dialog.isVisible()
    panel.close()
