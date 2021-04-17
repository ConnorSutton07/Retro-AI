import sys
from core import ui
from core import settings
from core import train

class Driver:
    def __init__(self) -> None:
        print()
        print("     +" + "-"*8 + "+")
        print("      RETRO AI")
        print("     +" + "-"*8 + "+")
        print()

    def run(self) -> None:
        modes = [
            ("Train AI", self._trainAI),
            ("Watch AI", self._watchAI),
            ("Exit", lambda: sys.exit())
        ]
        ui.runModes(modes)

    def _trainAI(self) -> None:
        modes = [
            ("Pretrained", self._runPretrained),
            ("New", self._runNew()),
            ("Back", self.run())
        ]
        ui.runModes(modes)

    def _runPretrained(self) -> None:
        print("NO")
        sys.exit()

    def _runNew(self) -> None:
        


    