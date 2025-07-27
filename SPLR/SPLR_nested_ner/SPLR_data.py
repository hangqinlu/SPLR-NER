def read_data(file):
    with open(file, encoding='utf-8') as f:
        all_data = f.read().split("\n")
    new_data = []

    for i in range(len(all_data)):
        if len(all_data[i]) < 3:
            continue
        new_data.append(eval(all_data[i]))
    return new_data


def build_type_index():
    with open(r"C:\Users\卢航青\PycharmProjects\pythonProject11\SPLR实验数据\SPLRtext", encoding='utf-8') as f:
        index_2_type = f.read().split("\n")
    type_2_index = {r: i for i, r in enumerate(index_2_type)}

    return type_2_index, index_2_type