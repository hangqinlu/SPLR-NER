import torch
from collections import Counter, defaultdict

def classify_nested_entities(entity_list):
    outer_set, inner_set = set(), set()
    n = len(entity_list)
    for i in range(n):
        e1, t1 = entity_list[i][0], entity_list[i][1]
        is_outer = False
        is_inner = False
        for j in range(n):
            if i == j: continue
            e2, t2 = entity_list[j][0], entity_list[j][1]
            if e2 != e1 and e2 in e1:
                is_outer = True
            # 内层判据：自己被别人完全包含
            if e2 != e1 and e1 in e2:
                is_inner = True
        if is_outer:
            outer_set.add((e1, t1))
        if is_inner:
            inner_set.add((e1, t1))
    return outer_set, inner_set

def multihot(length, pos):
    return [1 if i in pos else 0 for i in range(length)]


def get_correct_predictions(true_tensor, pred_tensor):
    correct_predictions = []

    # 遍历每个预测值
    for p in pred_tensor:
        if p.item() in true_tensor.tolist():
            correct_predictions.append(p.item())

    return correct_predictions

def get_entities(model, encode_text, text, mask, offset_mapping, index_2_rel):
    entities = []
    point = []
    pred = model.get_objs_for_specific_sub(encode_text)
    l_t = pred[0].T
    pred_mx_list = [mx.T for mx in pred[2:10]]
    delta_list = list(range(8))
    thresholds = [0.4] + [0.7] * 7

    for pred_mx, delta, thresh in zip(pred_mx_list, delta_list, thresholds):
        for j in range(pred_mx.size(0)):
            head_ids = torch.where(pred_mx[j] > thresh)[0]
            tail_ids = torch.where(l_t[0] > thresh)[0]
            for head_id in head_ids:
                tail_id = head_id + delta
                if tail_id in tail_ids:
                    if head_id < 0 or tail_id > len(offset_mapping):
                        print(
                            f"Skipping index: obj_head_id={head_id}, obj_tail_id={tail_id}, length={len(offset_mapping)}")
                        continue
                    try:
                        head_pos_id = offset_mapping[head_id][0]
                        tail_pos_id = offset_mapping[tail_id][1]
                        point .append([head_pos_id, tail_pos_id, j, delta + 1])
                    except IndexError:
                        print(f"Index out of range: obj_head_id={head_id}, obj_tail_id={tail_id}")
                        continue
    entity = [tuple(h) for h in point ]
    if not entity:
        return list((entities))
    else:

        for index, _ in enumerate(entity):
            pos_id = entity[index]
            head_id = pos_id[0]
            tail_id = pos_id[1]
            length = pos_id[-1]
            type = pos_id[2]
            object_text = text[head_id:tail_id]
            entities.append((object_text,index_2_rel[type],length))

    return entities


def report(model, encoded_text,pred_y, batch_text, batch_mask, index_2_rel):
    true_triple_list = batch_text["triple_list"]
    pre_triple_list = []
    correct_num, predict_num, gold_num, \
        correct_out_num, predict_out_num, gold_out_num, \
        correct_in_num, predict_in_num, gold_in_num, \
        nest_correct_num,nest_predict_num,nest_gold_num= (0,) * 12
    for i in range(len(pred_y[-1])):
        text = batch_text["text"][i]
        true_triple_item = true_triple_list[i]

        mask = batch_mask[i]
        offset_mapping = batch_text["offset_mapping"][i]
        pt= get_entities(model, encoded_text[i:i + 1], text, mask,offset_mapping, index_2_rel)

        if len(pt) != 0:
            pre_triple_list.append(pt)
            entity2types = defaultdict(set)
            for entity, typ in [(item[0], item[1]) for item in true_triple_item]:
                entity2types[entity].add(typ)
            multi_entity_set = set([entity for entity, types in entity2types.items() if len(types) > 1])
            pt_multi = [item for item in pt if item[0] in multi_entity_set]
            true_multi = [item for item in true_triple_item if item[0] in multi_entity_set]
            all_pred = Counter(pt)
            all_true = Counter(true_triple_item)

            true_outer, true_inner = classify_nested_entities(list(set(true_triple_item)))
            # 2. 分类预测
            pred_outer, pred_inner = classify_nested_entities(list(set(pt)))

            correct_out_num = len(true_outer & pred_outer)
            predict_out_num += len(pred_outer)
            gold_out_num += len(true_outer)

            correct_in_num = len(true_inner & pred_inner)
            predict_in_num += len(pred_inner)
            gold_in_num += len(pred_inner)

            correct_num = len(all_pred & all_true)
            predict_num+= len( all_pred)
            gold_num += len( all_true)

            nest_correct_num = len(set(pt_multi) & set(true_multi))
            nest_predict_num += len(set(pt_multi))
            nest_gold_num += len(set(true_multi))




    return (pre_triple_list, correct_num, predict_num, gold_num, true_triple_list,correct_out_num,predict_out_num,gold_out_num,correct_in_num,predict_in_num,gold_in_num
            ,nest_correct_num,nest_predict_num,nest_gold_num )

