from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from SPLR_nested_ner.SPLR_data import read_data,build_type_index
from SPLR_nested_ner.dataset import  RDataset
from SPLR_nested_ner.model import  RModel
from SPLR_nested_ner.eval import report
from SPLR_nested_ner.utils import compute_metrics
import yaml


def main():
    def load_config(path="config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config = load_config("config.yaml")

    train_data = read_data(config['data']['train_path'])
    test_data = read_data(config['data']['val_path'])
    type_2_index, index_2_type = build_type_index()
    model_dir = config['model']['backbone']
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    BERT = AutoModel.from_pretrained(model_dir)
    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    lr = config['train']['lr']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_f1_scores = []
    epoch_precisions = []
    epoch_recalls = []
    train_dataset = RDataset(train_data, type_2_index, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size, False, collate_fn=train_dataset.pro_batch_data)
    test_dataset = RDataset(test_data, type_2_index, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size, False, collate_fn=train_dataset.pro_batch_data)
    model = RModel(BERT, len(type_2_index)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        model.train()
        batch_loss = 0
        for bi, (batch_text, batch_mask, batch_mx) in enumerate(train_dataloader):
            input_mask = torch.tensor(batch_mask).to(device)
            input = torch.tensor(batch_text["input_ids"]).to(device)
            encoded_text, pred_y = model.forward(input, input_mask)
            mx_names = [
                "H_T_mx_tail", "H_T_mx_S", "H_T_mx_1", "H_T_mx_2", "H_T_mx_3",
                "H_T_mx_4", "H_T_mx_5", "H_T_mx_6", "H_T_mx_7", "H_T_mx_8"
            ]
            true_y = tuple(torch.tensor(batch_mx[name]).to(device) for name in mx_names)
            loss = model.loss_fun(true_y, pred_y, input_mask)
            for m in model.modules():
                if isinstance(m, nn.Conv1d):
                    for p in m.parameters():
                        p.requires_grad = False

            loss.backward()
            batch_loss += loss.item()

            opt.step()
            opt.zero_grad()
        model.eval()
        all_correct_num, all_predict_num, all_gold_num, \
            all_correct_out_num, all_out_pred_num, all_out_gold_num, \
            all_correct_in_num, all_in_pred_num, all_in_gold_num,\
            all_correct_nest_num,all_pred_nest_num,all_gold_nest_num= [0] * 12

        with torch.no_grad():
            for bi, (batch_text, batch_mask, batch_mx) in enumerate(test_dataloader):
                input_mask = torch.tensor(batch_mask).to(device)
                input = torch.tensor(batch_text["input_ids"]).to(device)
                encoded_text, pred_y = model.forward(input, input_mask)
                pre_list, correct_num, predict_num, gold_num, true_triple_list, correct_out_num, predict_out_num, gold_out_num, correct_in_num, predict_in_num, gold_in_num,nest_correct_num,nest_predict_num,nest_gold_num  = report(
                    model,
                    encoded_text,
                    pred_y,
                    batch_text,
                    batch_mask,
                    index_2_type
                )
                update_vars = {
                    'all_correct_num': correct_num,
                    'all_predict_num': predict_num,
                    'all_gold_num': gold_num,
                    'all_correct_nest_num': nest_correct_num,
                    'all_pred_nest_num': nest_predict_num,
                    'all_gold_nest_num': nest_gold_num,
                    'all_correct_out_num': correct_out_num,
                    'all_out_pred_num': predict_out_num,
                    'all_out_gold_num': gold_out_num,
                    'all_correct_in_num': correct_in_num,
                    'all_in_pred_num': predict_in_num,
                    'all_in_gold_num': gold_in_num
                }

                for var_name, value in update_vars.items():
                    globals()[var_name] += value

            compute_metrics(all_correct_num, all_predict_num, all_gold_num, "实体识别性能", epoch_f1_scores,
                            epoch_precisions, epoch_recalls)
            compute_metrics(all_correct_nest_num, all_pred_nest_num, all_gold_nest_num, "多义实体识别性能",
                            epoch_f1_scores, epoch_precisions, epoch_recalls)
            compute_metrics(all_correct_out_num, all_out_pred_num, all_out_gold_num, "嵌套外层识别性能",
                            epoch_f1_scores, epoch_precisions, epoch_recalls)
            compute_metrics(all_correct_in_num, all_in_pred_num, all_in_gold_num, "嵌套内层识别性能", epoch_f1_scores,
                            epoch_precisions, epoch_recalls)
    torch.save(model, "my_full_model.pt")




if __name__ == '__main__':
    main()