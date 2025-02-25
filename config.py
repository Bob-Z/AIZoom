import json
import sys

data = {}

with open("config.json", "r") as file:
    data = json.load(file)

def read_command_line():
    skip_first = True
    for arg in sys.argv:
        if skip_first:
            skip_first = False
            continue
        param = arg.split('=')
        if param[0] not in data:
            print("Unknown parameter:", param[0])
            sys.exit(0)
        print("Parameter", param[0], "set to", param[1])
        if type(data[param[0]]) is bool:
            if param[1].lower() in ['true', '1', 't', 'y', 'yes']:
                param[1] = True
            else:
                param[1] = False
        elif type(data[param[0]]) is float:
            param[1] = float(param[1])
        elif type(data[param[0]]) is int:
            param[1] = int(param[1])

        data[param[0]] = param[1]

