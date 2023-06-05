import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import glue_compute_metrics as compute_metrics
from transformers.optimization import AdamW
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix

# Bert Tokenization:
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
sent = 'This is my dog. His name is Jack.'
sent2 = 'He loves playing.'
inputs = tokenizer.encode_plus(text=sent, text_pair=sent2, add_special_tokens=True, max_length=15)


# Model class:
class BERTBASECLASSIFIER(nn.Module):
    def __init__(self, bert_type, num_labels):
        super(BERTBASECLASSIFIER, self).__init__()
        self.bert_type = bert_type
        self.num_labels = num_labels
        self.bert = transformers.BertForSequenceClassification.from_pretrained(
            self.bert_type,
            num_labels=self.num_labels)

    def forward(self, ids, mask_ids, token_ids, label):
        outputs = self.bert(
            input_ids=ids,
            attention_mask=mask_ids,
            token_type_ids=token_ids,
            labels=label)
        loss, logits = outputs[:2]
        return logits


# Dataset class:
class BertDatasetModule(Dataset):
    def __init__(self, tokenizer, input_sent, max_length, target):
        self.input_seq = input_sent
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target = target

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_ = self.input_seq[idx]
        inputs = self.tokenizer.encode_plus(text=input_, add_special_tokens=True, max_length=self.max_length)
        ids = inputs['input_ids']
        mask_ids = inputs['attention_mask']
        token_ids = inputs['token_type_ids']

        padding_len = self.max_length - len(ids)
        ids = ids + ([0] * padding_len)
        mask_ids = mask_ids + ([0] * padding_len)
        token_ids = token_ids + ([0] * padding_len)

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask_ids, dtype=torch.long),
                'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
                'target': torch.tensor(self.target[idx], dtype=torch.int16)}


# Defining loss:
def loss_func(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


# Model training:
def train_loop(dataloader, model, optimizer, device, max_grad_norm, scheduler=None):
    model.train()
    for bi, d in enumerate(tqdm(dataloader, desc="Iteration")):
        ids = d['ids']
        mask_ids = d['mask']
        token_ids = d['token_type_ids']
        target = d['target']

        ids = ids.to(device, dtype=torch.long)
        mask_ids = mask_ids.to(device, dtype=torch.long)
        token_ids = token_ids.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(ids, mask_ids, token_ids, target)
        loss = loss_func(output, target)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if bi % 100 == 0:
            print(f"bi: {bi}, loss: {loss}")

acc_list = []
# Model evaluation:
def eval_loop(dataloader, model, device):
    global acc_list
    model.eval()
    preds = None
    out_label_ids = None
    eval_loss = 0.0
    eval_steps = 0

    for bi, d in enumerate(dataloader):
        ids = d['ids']
        mask_ids = d['mask']
        token_ids = d['token_type_ids']
        target = d['target']

        ids = ids.to(device, dtype=torch.long)
        mask_ids = mask_ids.to(device, dtype=torch.long)
        token_ids = token_ids.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(ids, mask_ids, token_ids, target)
            loss = loss_func(output, target)
            eval_loss += loss.mean().item()

        eval_steps += 1
        if preds is None:
            preds = output.detach().cpu().numpy()
            out_label_ids = target.detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, target.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / eval_steps
    preds = np.argmax(preds, axis=1)

    conf_matrix = confusion_matrix(out_label_ids, preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel()
    print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')
    acc = (tp+tn)/(tp+tn+fn+fp)
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    print("A: {}, P: {}, R: {}".format(acc, pre, rec))
    acc_list.append(acc)
    return eval_loss


def dataset_details(df):
    print("Dataset preview")
    print(df.head(5))
    print("label count:")
    print(df.groupby([0]).count())


def run():
    MAX_SEQ_LENGTH = 128
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    NUM_TRAIN_EPOCHS = 20
    NUM_LABELS = 2
    BERT_TYPE = "bert-base-uncased"
    max_grad_norm = 1.0

    train_df = pd.read_csv('train.csv', header=None)
    test_df = pd.read_csv('test.csv', header=None)
    train_df[0] = (train_df[0] == 2).astype(int)
    test_df[0] = (test_df[0] == 2).astype(int)

    dataset_details(train_df)
    dataset_details(test_df)

    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_TYPE)
    train_dataset = BertDatasetModule(
        tokenizer=tokenizer,
        input_sent=train_df[1],
        max_length=MAX_SEQ_LENGTH,
        target=train_df[0]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    eval_dataset = BertDatasetModule(
        tokenizer=tokenizer,
        input_sent=test_df[1],
        max_length=MAX_SEQ_LENGTH,
        target=test_df[0]
    )

    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("Pytorch device: {}".format(device))

    model = BERTBASECLASSIFIER(BERT_TYPE, NUM_LABELS).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)

    NUM_TRAIN_STEPS = int(len(train_dataset) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    scheduler = transformers.get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        # num_training_steps=NUM_TRAIN_STEPS,
        last_epoch=-1)

    for epoch in trange(NUM_TRAIN_EPOCHS):
        print("Train epoch: {}".format(epoch))
        train_loop(train_dataloader, model, optimizer, device, max_grad_norm, scheduler)

        res = eval_loop(eval_dataloader, model, device)
        print(res)
        plt.plot(acc_list)

    res = eval_loop(eval_dataloader, model, device)
    print(res)


if __name__ == '__main__':
    run()
