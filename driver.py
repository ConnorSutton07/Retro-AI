import sys
import os 
from core import ui
from core import train
from retro.scripts.import_path import main as import_roms

class Driver:
    def __init__(self) -> None:

        self.paths = {}
        self.paths["current"] = os.getcwd()
        self.paths["models"] = os.path.join(self.paths["current"], "models")
        self.paths["ROMs"] = os.path.join(self.paths["current"], "ROMs")
        

        print()
        print("     +" + "-"*8 + "+")
        print("      RETRO AI")
        print("     +" + "-"*8 + "+")
        print()

    def run(self) -> None:
        modes = [
            ("Train AI", self._trainAI),
            ("Watch AI", self._viewPretrained),
            ("Import ROMs", self.importROMs),
            ("Exit", lambda: sys.exit())
        ]
        ui.runModes(modes)

    def _trainAI(self) -> None:
        modes = [
            ("Pretrained", self._runPretrained),
            ("New", self._runNew),
            ("Back", self.run)
        ]
        ui.runModes(modes, "Select training option:")

    def _runPretrained(self) -> None:
        modelPath = self.getPretrainedModel()
        session_name = input("Name for session: ")
        num_episodes = input("Number of training episodes: ")
        session_path = os.path.join(self.paths["models"], session_name)
        os.mkdir(session_path)
        if modelPath != None:
            train.run(training_mode=True,
                        pretrained=True,
                        num_episodes=int(num_episodes),
                        save_path=session_path,
                        load_path=modelPath)

    def _viewPretrained(self) -> None:
        modelPath = self.getPretrainedModel()
        if modelPath != None:
            train.run(training_mode=False,
                        pretrained=True,
                        num_episodes=500,
                        save_path=None,
                        load_path=modelPath)

    def getPretrainedModel(self) -> str:
        models = os.listdir(self.paths["models"])
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
                return os.path.join(self.paths["models"], models[index])

        return None
        
    def _runNew(self) -> None:
        session_name = input("Name for session: ")
        num_episodes = input("Number of training episodes: ")
        session_path = os.path.join(self.paths["models"], session_name)
        os.mkdir(session_path)
        train.run(training_mode = True,
                  pretrained = False,
                  num_episodes = int(num_episodes),
                  save_path = session_path)

    def importROMs(self) -> None:
        print("Checking ROMs directory...")
        #import_roms(self.paths["ROMs"])
        sys.argv = [sys.argv[0], self.paths["ROMs"]]
        import_roms()



    