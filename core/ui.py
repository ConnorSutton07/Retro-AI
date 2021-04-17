"""
User Interaction

"""
import os
import json

def getValidInput(msg: str,
                  dtype: any = str,
                  lower: float = None,
                  upper: float = None,
                  valid: set = None,
                  isValid: callable = None,
                  end = None) -> any:

    print(msg)
	while True:
		try:
			choice = dtype(input("\nChoice: "))
		except ValueError:  # if type can't be properly converted into dtype
			continue
		if (lower is None or choice >= lower) and \
				(upper is None or choice <= upper) and \
				(valid is None or choice in valid) and \
				(isValid is None or isValid(choice)):
			if end is not None:
				print("", end=end)
			return choice

def getSelection(*args, msg: str = "Choose item:", end = None, **kwargs) -> tuple:
    for i, item in enumerate(args):
        msg += "\n    " + str(i + 1) + ") " + str(item)
    i = getValidInput(msg, dtype=int, lower=1, upper=len(args), end=end, **kwargs) - 1
    return i, args[i]

def hasExtension(name: str, ext: str) -> bool:
    return (len(name) > len(ext) + 1 and name[-1 * len(ext):] == ext)

def saveToJSON(data, path, indent=4) -> None:
    if not hasExtension(os.path.basename(path), "json"):
        path += ".json"
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

def runModes(modes: list) -> None:
    mode = None
    modes, callbacks = list(zip(*modes))
    while mode != "Exit":
        index, mode = getSelection(*modes, msg="Select mode:")
        callbacks[index]()
        print()
