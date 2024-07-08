import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm, trange
from old_code.create_clone_bcb import create_ast, create_gmn_data, create_separate_graph
import models

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default="gcj")
parser.add_argument("--graph_mode", default="ast_and_next")
parser.add_argument("--next_sib", default=False)
parser.add_argument("--if_edge", default=False)
parser.add_argument("--while_edge", default=False)
parser.add_argument("--for_edge", default=False)
parser.add_argument("--block_edge", default=False)
parser.add_argument("--next_token", default=False)
parser.add_argument("--next_use", default=False)
parser.add_argument("--data_setting", default="0")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()

device = torch.device("cuda:0")
# device=torch.device('cpu')
astdict, vocab_len, vocab_dict = create_ast()
tree_dict = create_separate_graph(
    astdict,
    vocab_len,
    vocab_dict,
    device,
    mode=args.graph_mode,
    next_sib=args.next_sib,
    if_edge=args.if_edge,
    while_edge=args.while_edge,
    for_edge=args.fore_dge,
    block_edge=args.block_edge,
    next_token=args.next_token,
    next_use=args.next_use,
)
train_data, valid_data, test_data = create_gmn_data(
    args.data_setting, tree_dict, vocab_len, vocab_dict, device
)
print(len(train_data))
num_layers = int(args.num_layers)
model = models.GGNN(
    vocab_len, embedding_dim=100, num_layers=num_layers, device=device
).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CosineEmbeddingLoss()
criterion_2 = nn.MSELoss()


def create_batches(data):
    batches = [
        data[graph : graph + args.batch_size]
        for graph in range(0, len(data), args.batch_size)
    ]
    return batches


def test(dataset):
    tp, tn, fp, fn = 0, 0, 0, 0
    results = []
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        x_1, x_2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2 = data
        x_1 = torch.tensor(x_1, dtype=torch.long, device=device)
        x_2 = torch.tensor(x_2, dtype=torch.long, device=device)
        edge_index_1 = torch.tensor(edge_index_1, dtype=torch.long, device=device)
        edge_index_2 = torch.tensor(edge_index_2, dtype=torch.long, device=device)
        if edge_attr_1 != None:
            edge_attr_1 = torch.tensor(edge_attr_1, dtype=torch.long, device=device)
            edge_attr_2 = torch.tensor(edge_attr_2, dtype=torch.long, device=device)
        data_1 = [x_1, edge_index_1, edge_attr_1]
        data_2 = [x_2, edge_index_2, edge_attr_2]
        prediction_1 = model(data_1)
        prediction_2 = model(data_2)
        output = F.cosine_similarity(prediction_1, prediction_2)
        results.append(output.item())
        prediction = torch.sign(output).item()
        if prediction > args.threshold and label.item() == 1:
            tp += 1
        if prediction <= args.threshold and label.item() == -1:
            tn += 1
        if prediction > args.threshold and label.item() == -1:
            fp += 1
        if prediction <= args.threshold and label.item() == 1:
            fn += 1
    print(tp, tn, fp, fn)
    p, r, f1 = 0.0, 0.0, 0.0
    if tp + fp == 0:
        return print("precision is none")
    p = tp / (tp + fp)
    if tp + fn == 0:
        return print("recall is none")
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print("precision", p)
    print("recall", r)
    print("F1", f1)
    return results


epochs = trange(args.num_epochs, leave=True, desc="Epoch")
for epoch in epochs:  # without batching
    print(epoch)
    batches = create_batches(train_data)
    total_loss = 0.0
    main_index = 0.0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        optimizer.zero_grad()
        batch_loss = 0
        for data, label in batch:
            label = torch.tensor(label, dtype=torch.float, device=device)
            x_1, x_2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2 = data
            x_1 = torch.tensor(x_1, dtype=torch.long, device=device)
            x_2 = torch.tensor(x_2, dtype=torch.long, device=device)
            edge_index_1 = torch.tensor(edge_index_1, dtype=torch.long, device=device)
            edge_index_2 = torch.tensor(edge_index_2, dtype=torch.long, device=device)
            if edge_attr_1 != None:
                edge_attr_1 = torch.tensor(edge_attr_1, dtype=torch.long, device=device)
                edge_attr_2 = torch.tensor(edge_attr_2, dtype=torch.long, device=device)
            data_1 = [x_1, edge_index_1, edge_attr_1]
            data_2 = [x_2, edge_index_2, edge_attr_2]
            prediction_1 = model(data_1)
            prediction_2 = model(data_2)
            cos_sim = F.cosine_similarity(prediction_1, prediction_2)
            batch_loss = batch_loss + criterion_2(cos_sim, label)
        batch_loss.backward(retain_graph=True)
        optimizer.step()
        loss = batch_loss.item()
        total_loss += loss
        main_index = main_index + len(batch)
        loss = total_loss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
    dev_results = test(valid_data)
    dev_file = open(
        "ggnnbcbresult/" + args.graph_mode + "_dev_epoch_" + str(epoch + 1), mode="w"
    )
    for res in dev_results:
        dev_file.write(str(res) + "\n")
    dev_file.close()
    # test(testdata)
    testresults = test(test_data)
    resfile = open(
        "ggnnbcbresult/" + args.graph_mode + "_epoch_" + str(epoch + 1), mode="w"
    )
    for res in testresults:
        resfile.write(str(res) + "\n")
    resfile.close()

    # for start in range(0, len(traindata), args.batch_size):
    # batch = traindata[start:start+args.batch_size]
    # epochs.set_description("Epoch (Loss=%g)" % round(loss,5))


"""for batch in trainloder:
    batch=batch.to(device)
    print(batch)
    quit()
    time_start=time.time()
    model.forward(batch)
    time_end=time.time()
    print(time_end-time_start)
    quit()"""
