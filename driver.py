import sys
import os 
from core import ui
from core import train
from retro.scripts.import_path import main as import_roms
from retro.data import get_known_hashes, groom_rom

class Driver:
    def __init__(self) -> None:

        self.paths = {}
        self.paths["current"] = os.getcwd()
        self.paths["models"] = os.path.join(self.paths["current"], "models")
        self.paths["ROMs"] = os.path.join(self.paths["current"], "ROMs")
        self.algorithms = ["DQN", "PPO"]        

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
        ui.runModes(modes, "Mode:")

    def _runNew(self) -> None:
        game = self.getROM()
        print(game)
        if game is None: return
        algorithm    = self.chooseTrainingAlgorithm()
        if algorithm is None: return
        session_name = input("Name for session: ")
        num_episodes = input("Number of training episodes: ")
        session_path = os.path.join(self.paths["models"], session_name)
        os.mkdir(session_path)
        train.run(game = game,
                  alg = algorithm,
                  training_mode = True,
                  pretrained = False,
                  num_episodes = int(num_episodes),
                  save_path = session_path)

    def _runPretrained(self) -> None:
        game = self.getROM()
        if game is None: return
        modelPath = self.getPretrainedModel()
        if modelPath is None: return
        session_name = input("Name for session: ")
        num_episodes = input("Number of training episodes: ")
        session_path = os.path.join(self.paths["models"], session_name)
        os.mkdir(session_path)
        train.run(game = game,
                        training_mode=True,
                        pretrained=True,
                        num_episodes=int(num_episodes),
                        save_path=session_path,
                        load_path=modelPath)

    def _viewPretrained(self) -> None:
        game = self.getROM()
        if game is None: return
        modelPath = self.getPretrainedModel()
        if modelPath != None:
            train.run(game = game,
                      training_mode=False,
                      pretrained=True,
                      num_episodes=500,
                      save_path=None,
                      load_path=modelPath)

    def chooseTrainingAlgorithm(self) -> str:
        """
        Allows user to choose which RL algorithm will be used.
        Options: DQN, PPO

        """
        num_algorithms = len(self.algorithms)
        msg = "Select training algorithm:"
        for i, algorithm in enumerate(self.algorithms, start = 1):
            msg += "\n    " + str(i) + ") " + algorithm
        backIndex = num_algorithms + 1
        msg += "\n    " + str(backIndex) + ") Back"
        index = ui.getValidInput(msg, dtype=int, valid=range(1, num_algorithms + 2)) - 1
        if (index != backIndex - 1):
            return self.algorithms[index]
        return None

    def getPretrainedModel(self) -> str:
        models = os.listdir(self.paths["models"])
        num_models = len(models)
        
        if num_models == 0:
            print("\n No saved models.")
        else:
            print()
            msg = "Select model to use:"
            for i, model in enumerate(models, start=1):
                msg += "\n    " + str(i) + ") " + str(model)
            backIndex = num_models + 1
            msg += "\n    " + str(backIndex) + ") Back"
            index = ui.getValidInput(msg, dtype=int, valid=range(1, num_models + 2)) - 1

            if (index != backIndex - 1):
                return os.path.join(self.paths["models"], models[index])
        return None

    def importROMs(self) -> None:
        """
        Uses Gym-Retro's import module which reads
        the ROM path via command line argument. In 
        order to use Retro without making alterations,
        we can save/mutate/restore sys.argv

        """
        print("Checking ROMs directory...")
        old_argv = sys.argv
        sys.argv = [sys.argv[0], self.paths["ROMs"]]
        import_roms()
        sys.argv = old_argv

    def getROM(self) -> str:
        """ 
        Displays all (SNES) ROMs in ROMs directory and
        allows user to select one. At the moment, only
        SNES files are supported.
        Note: This method may display unimported ROMS

        """
        files = os.listdir(self.paths["ROMs"])
        snes_games = [file for file in files if ui.hasExtension(file, 'smc') or ui.hasExtension(file, 'sfc')]
        num_games = len(snes_games)
        if num_games == 0:
            print("\n No ROMs found. Make sure any ROMs are in the correct directory. ")
        else:
            print()
            known_hashes = get_known_hashes()
            games = []
            msg = "Select game:"
            for i, rom in enumerate(snes_games, start=1):
                with open(os.path.join(self.paths["ROMs"], rom), "rb") as f:
                    _, hash = groom_rom(rom, f)
                    if hash in known_hashes:
                        game, _, _ = known_hashes[hash]
                        games.append(game)
                        msg += "\n    " + str(i) + ") " + game
                #msg += "\n\t" + str(i) + ") " + str(rom)
            backIndex = num_games + 1
            msg += "\n    " + str(backIndex) + ") Back"
            index = ui.getValidInput(msg, dtype=int, valid=range(1, num_games + 2)) - 1

            if (index != backIndex - 1):
                return games[index]
        return None
        
        #print(snes_files)

        



    