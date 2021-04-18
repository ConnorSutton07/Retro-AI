import sys
import os 
from core import ui
from core import train

class Driver:
    def __init__(self) -> None:

        self.paths = {}
        self.paths["current"] = os.getcwd()
        self.paths["data"] = os.path.join(self.paths["current"], "data")
        self.paths["trained"] = os.path.join(self.paths["data"], "trained")
        

        print()
        print("     +" + "-"*8 + "+")
        print("      RETRO AI")
        print("     +" + "-"*8 + "+")
        print()

    def run(self) -> None:
        modes = [
            ("Train AI", self._trainAI),
            ("Watch AI", self._viewPretrained, False),
            ("Exit", lambda: sys.exit())
        ]
        ui.runModes(modes)

    def _trainAI(self) -> None:
        modes = [
            ("Pretrained", self._runPretrained, True),
            ("New", self._runNew),
            ("Back", self.run)
        ]
        ui.runModes(modes, "Select training option:")

    def _runPretrained(self) -> None:
        modelPath = self.getPretrainedModel()
        if modelPath != None:
            train.run(training_mode=True,
                        pretrained=True,
                        num_episodes=500,
                        session_name="test",
                        load_path=modelPath)

    def _viewPretrained(self) -> None:
        modelPath = self.getPretrainedModel()
        if modelPath != None:
            train.run(training_mode=False,
                        pretrained=True,
                        num_episodes=500,
                        session_name="test",
                        load_path=modelPath)

    def getPretrainedModel(self) -> str:
        models = os.listdir(self.paths["trained"])
        num_models = len(models)
        
        if num_models == 0:
            print("\n No saved models.")
        else:
            print()
            msg = "Select model to use:"
            for i, model in enumerate(models, start=1):
                msg += "\n\t" + str(i) + ") " + str(model)
            backIndex = num_models + 1
            msg += "\n\t" + str(backIndex) + ") Back"
            index = ui.getValidInput(msg, dtype=int, valid=range(1, num_models+2)) - 1

            if (index != backIndex - 1):
                return os.path.join(self.paths["trained"], models[index])

        return None
        
    def _runNew(self) -> None:
        pass


    