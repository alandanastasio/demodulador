"""Transiciones de modo sin entregar IQ a la cadena DSP."""

from functools import wraps
from inspect import signature

from .hackrf_handler import HackRFHandler


def receiver_transition(method):
    """Pausa HackRF durante una transición, también cuando hay llamadas anidadas."""
    no_arguments = len(signature(method).parameters) == 1

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        # QAction.triggered emite un bool. Qt lo descarta para métodos sin
        # parámetros, pero envía el valor al wrapper con *args.
        if no_arguments and len(args) == 1 and isinstance(args[0], bool) and not kwargs:
            args = ()

        if not isinstance(self.radio, HackRFHandler):
            return method(self, *args, **kwargs)

        depth = getattr(self, '_receiver_transition_depth', 0)
        if depth == 0:
            self._receiver_was_running = self.radio.is_running
            if self._receiver_was_running:
                self.radio.stop_rx()
        self._receiver_transition_depth = depth + 1
        succeeded = False
        try:
            result = method(self, *args, **kwargs)
            succeeded = True
            return result
        finally:
            self._receiver_transition_depth -= 1
            if self._receiver_transition_depth == 0 and self._receiver_was_running and succeeded:
                self.radio.start_rx()

    return wrapped
