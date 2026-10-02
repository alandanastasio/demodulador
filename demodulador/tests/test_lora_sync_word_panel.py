"""El panel LoRa cuenta e indica actividad de cada sync word por separado."""


def test_lora_sync_word_cards_count_and_blink_independently(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ui_builder import LoRaSyncWordPanel

    app = QApplication.instance() or QApplication([])
    panel = LoRaSyncWordPanel()
    panel.show()

    panel.record(0x12)
    panel.record(0x34)
    panel.record(0x12)

    assert list(panel.entries) == [0x12, 0x34]
    assert panel.entries[0x12]['count'] == 2
    assert panel.entries[0x34]['count'] == 1
    assert panel.entries[0x12]['counter'].text() == 'Paquetes: 2'
    assert panel.scroll_area.height() == 98
    assert '#4be16b' in panel.entries[0x12]['led'].styleSheet()

    QTest.qWait(120)
    panel.record(0x12)
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
