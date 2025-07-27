from torch.utils.data import DataLoader, Dataset
MAX_ENTITY_LEN = 8
class RDataset(Dataset):
    def __init__(self, all_data, rel_2_index, tokenizer):
        self.all_data = all_data
        self.rel_2_index = rel_2_index
        self.tokenizer = tokenizer

    def parase_info(self, info):
        text = info["sentence"]
        input_ids = info["input_ids"]
        dct = {
            "text": text,
            "input_ids": input_ids,
            "offset_mapping": info["offset_mapping"],
            "head_ids": [],
            "tail_ids": [],
            "triple_list": [],
            "type_id_list": [],
            "pos_ids": [],
            "head_id":[]
        }
        if len(info["entities"]) != 0:
            for a in info["entities"]:
                entity = a["text"]
                entity_type = a["entity_type"].split(":")[0]
                sub_token = self.tokenizer(entity, add_special_tokens=False)["input_ids"]
                max_len = len(sub_token)
                dct["triple_list"].append((entity,entity_type,max_len))
                sub_pos_id = self.find_pos_id(input_ids, sub_token)
                if sub_pos_id is None:
                    continue
                for a in sub_pos_id:
                    head_id = a[0]
                    tail_id = a[1]
                    dct["pos_ids"].append(a)
                    rel_id = self.rel_2_index[entity_type]
                    dct["type_id_list"].append((
                        [head_id, tail_id],
                        rel_id , max_len
                    ))

        return dct

    def find_pos_id(self, source, element):
        pos_id = []
        for h_i in range(len(source)):
            t_i = h_i + len(element)
            if source[h_i:t_i] == element:
                pos_id.append([h_i, t_i - 1])

        return pos_id

    def __getitem__(self, index):
        info = self.all_data[index]
        text = info['sentence']
        token = self.tokenizer(text, return_offsets_mapping=True)

        input_ids = token["input_ids"]
        offset_mapping = token['offset_mapping']

        info["input_ids"] = input_ids
        info["offset_mapping"] = offset_mapping
        return self.parase_info(info)

    def multihot(self, length, pos):
        return [1 if i in pos else 0 for i in range(length)]

    def pro_batch_data(self, batch_data):
        batch_data = sorted(batch_data, key=lambda x: len(x["input_ids"]), reverse=True)
        max_len = len(batch_data[0]["input_ids"])
        batch_text = {
            "text": [],
            "input_ids": [],
            "offset_mapping": [],
            "triple_list": []
        }
        batch_mask = []

        batch_mx = {"H_T_mx_tail": [],"H_T_mx_S":[]}
        for k in range(1, MAX_ENTITY_LEN + 1):
            batch_mx[f"H_T_mx_{k}"] = []
        for item in batch_data:
            # 创建 tail 和所有长度的实体优化张量
            H_T_mx_S=[[0] * MAX_ENTITY_LEN for _ in range(max_len)]
            H_T_mx_tail = [[0] for _ in range(max_len)]
            H_T_mx_list = [
                [[0] * len(self.rel_2_index) for _ in range(max_len)]
                for _ in range(MAX_ENTITY_LEN)
            ]
            input_ids = item["input_ids"]
            item_len = len(input_ids)
            pad_len = max_len - item_len
            input_ids = input_ids + [0] * pad_len
            mask = [1] * item_len + [0] * pad_len

            for triple in item.get("type_id_list", []):
                length = triple[-1]
                head_id, tail_id = triple[0]
                rel_id = triple[1]
                if length <= MAX_ENTITY_LEN:
                   H_T_mx_S[head_id][length-1] = 1
                H_T_mx_tail[tail_id][0] = 1
                if 1 <= length <= MAX_ENTITY_LEN:
                    H_T_mx_list[length - 1][head_id][rel_id] = 1

            batch_mx["H_T_mx_tail"].append(H_T_mx_tail)
            for k in range(MAX_ENTITY_LEN):
                batch_mx[f"H_T_mx_{k + 1}"].append(H_T_mx_list[k])

            batch_mx["H_T_mx_S"].append(H_T_mx_S)
            batch_text["text"].append(item["text"])
            batch_text["input_ids"].append(input_ids)
            batch_text["offset_mapping"].append(item["offset_mapping"])
            batch_text["triple_list"].append(item["triple_list"])

            batch_mask.append(mask)

        return batch_text, batch_mask, batch_mx

    def __len__(self):
        return len(self.all_data)
