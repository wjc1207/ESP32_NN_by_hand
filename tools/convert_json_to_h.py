import json
import numpy as np

# Assume that there is a "model.json" file
file_path = 'model.json'

# read json file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

with open("model.h", "w", encoding="utf-8") as file:
    for key, item in data.items():
        np_array_item = np.array(item)
        shape = np_array_item.shape
        file.write("static float ")
        file.write(f"%s"%(key.replace('.', '_')))
        for i in range(len(shape)):
            file.write("[")
            file.write(f"%d"%(shape[i]))
            file.write("]")
        file.write(" = ")

        item = str(item)
        item = item.replace('[', '{').replace(']', '}')
        file.write(item)

        file.write(";\n")

        print(key)