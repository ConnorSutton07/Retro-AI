"""
User interaction.
"""
import os
import json

def getValidInput(
				  msg: str,
				  dtype: any = str,
				  lower: float = None, upper: float = None,
				  valid: set = None,
				  isValid: callable = None,
				  end=None
				  ) -> any:
	"""
	Gets input from user constrained by parameters.
	Parameters
	----------
	msg: str
		Message to print out to user requesting input
	dtype: any, default=str
		Type that input will get converted to
	lower: float, optional
		Numerical lower bound
	upper: float, optional
		Numerical upper bound
	valid: set, optional
		Set of possible valid inputs
	isValid: callable, optional
		Function returning bool to determine if input is valid
	Returns
	-------
	any: valid user input
	"""
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
	
def getSelection(*args, msg: str = "Choose item:", end=None, **kwargs) -> tuple:
	for i, item in enumerate(args):
		msg += "\n    " + str(i + 1) + ") " + str(item)
		
	i = getValidInput(msg, dtype=int, lower=1, upper=len(args), end=end, **kwargs) - 1
	return i, args[i]
	
def hasExtension(name, ext):
	return (len(name) > len(ext) + 1 and name[-1 * len(ext):] == ext)
	
def saveToJSON(data, path, indent=4) -> None:
	"""
	Saves game environment to .json file in .../replay folder.
	Parameters
	----------
	name: str
		File name for saved game file
	"""
	if not hasExtension(os.path.basename(path), "json"):
		path += ".json"
	with open(path, "w") as f:
		json.dump(data, f, indent=indent)
		
def runModes(modes, msg: str = ""):
	mode = None
	modes, callbacks = list(zip(*modes))
	while mode != "Exit":
		index, mode = getSelection(*modes, msg=msg)
		callbacks[index]()
		print()

def checkSave(data, callback, msg="Save?"):
	"""Checks to see if user wants to save."""
	index, _ = getSelection("Yes", "No", msg=msg)
	if not index:
		print()
		name = getValidInput("Name?")
		callback(data, name)
		
def formatTime(time):
	return str(int(time//3600)) + " hrs " + str(int((time//60)%60)) + " mins " + str(int(time%60)) + " secs"
