def compute_metrics(correct, pred, gold, desc, f1_list, p_list, r_list):
    precision = correct / (pred + 1e-8)
    recall = correct / (gold + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_list.append(f1)
    p_list.append(precision)
    r_list.append(recall)
    print(f"{desc} F1: {f1:.3f}")