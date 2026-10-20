"""El callback de salida conserva muestras parciales y prioriza audio reciente."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np


class Button:
    def __init__(self):
        self.checked = True

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = value

    def setStyleSheet(self, _style):
        pass


class Stream:
    def __init__(self, **kwargs):
        self.callback = kwargs['callback']
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        pass


def make_manager(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=Stream))
    module = importlib.import_module('audio_manager')
    ui = SimpleNamespace(audio_l_btn=Button(), audio_r_btn=Button())
    manager = module.AudioManager(ui, {})
    manager.toggle_audio()
    return manager


def play(manager, frames):
    output = np.empty((frames, 2), dtype=np.float32)
    manager.audio_stream.callback(output, frames, None, None)
    return output


def test_partial_audio_chunk_is_played_before_silence(monkeypatch):
    manager = make_manager(monkeypatch)
    manager.enqueue_audio(np.ones((4800, 2), dtype=np.float32))
    manager.enqueue_audio(np.full((4800, 2), 2, dtype=np.float32))
    manager.enqueue_audio(np.full((4800, 2), 3, dtype=np.float32))

    for _ in range(4):
        np.testing.assert_array_equal(play(manager, 1024), 1)
    tail = play(manager, 1024)

    np.testing.assert_array_equal(tail[:704], 1)
    np.testing.assert_array_equal(tail[704:], 2)
    np.testing.assert_array_equal(play(manager, 1024), 2)
    manager.deactivate()


def test_audio_waits_for_two_reserve_blocks(monkeypatch):
    manager = make_manager(monkeypatch)
    manager.enqueue_audio(np.ones((4800, 2), dtype=np.float32))
    np.testing.assert_array_equal(play(manager, 1024), 0)
    manager.enqueue_audio(np.ones((4800, 2), dtype=np.float32))
    np.testing.assert_array_equal(play(manager, 1024), 0)
    manager.enqueue_audio(np.ones((4800, 2), dtype=np.float32))
    np.testing.assert_array_equal(play(manager, 1024), 1)
    manager.deactivate()


def test_audio_counts_real_queue_underrun_after_start(monkeypatch):
    manager = make_manager(monkeypatch)
    for _ in range(3):
        manager.enqueue_audio(np.ones((4800, 2), dtype=np.float32))

    np.testing.assert_array_equal(play(manager, 14400), 1)
    np.testing.assert_array_equal(play(manager, 1024), 0)
    assert manager.flow_stats()['audio_underruns'] == 1
    assert manager.flow_stats()['audio_underrun_ms'] == 1024 / 48
    manager.deactivate()


def test_live_queue_drops_oldest_block_when_full(monkeypatch):
    manager = make_manager(monkeypatch)
    for value in range(1, 8):
        manager.enqueue_audio(np.full((4800, 2), value, dtype=np.float32))

    assert manager.state['audio_queue'].qsize() == 6
    np.testing.assert_array_equal(play(manager, 1024), 2)
    assert manager.flow_stats()['dropped_audio_blocks'] == 1
    manager.deactivate()
    assert manager.audio_stream is None
    assert manager.state['audio_queue'].empty()
