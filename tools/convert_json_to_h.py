import json
import numpy as np

# 假设有一个名为data.json的JSON文件
file_path = 'model.json'

# 读取JSON文件
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
        print(item)

        file.write(";\n")