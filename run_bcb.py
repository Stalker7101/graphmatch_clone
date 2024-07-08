import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm, trange
from create_clone_bcb import create_ast, create_gmn_data, create_separate_graph
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
parser.add_argument("--data_setting", default="11")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
print("CUDA", torch.cuda.is_available())


ast_dict, vocab_len, vocab_dict = create_ast()
tree_dict = create_separate_graph(
    ast_dict,
    vocab_len,
    vocab_dict,
    device,
    mode=args.graph_mode,
    next_sib=args.next_sib,
    if_edge=args.if_edge,
    while_edge=args.while_edge,
    for_edge=args.for_edge,
    block_edge=args.block_edge,
    next_token=args.next_token,
    next_use=args.next_use,
)
train_data, valid_data, test_data = create_gmn_data(
    args.data_setting, tree_dict, vocab_len, vocab_dict, device
)
print(len(train_data))
num_layers = int(args.num_layers)
model = models.GMNnet(
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


def predict(data):
    x_1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2 = data
    x_1 = torch.tensor(x_1, dtype=torch.long, device=device)
    x2 = torch.tensor(x2, dtype=torch.long, device=device)
    edge_index_1 = torch.tensor(edge_index_1, dtype=torch.long, device=device)
    edge_index_2 = torch.tensor(edge_index_2, dtype=torch.long, device=device)
    if edge_attr_1 != None:
        edge_attr_1 = torch.tensor(edge_attr_1, dtype=torch.long, device=device)
        edge_attr_2 = torch.tensor(edge_attr_2, dtype=torch.long, device=device)
    data = [x_1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2]
    return model(data)


def validate(dataset):
    tp, tn, fp, fn = 0, 0, 0, 0
    results = []
    for data, label in dataset:
        prediction = predict(data)
        output = F.cosine_similarity(prediction[0], prediction[1])
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
            prediction = predict(data)
            cos_sim = F.cosine_similarity(prediction[0], prediction[1])
            batch_loss += criterion_2(cos_sim, torch.tensor(label, dtype=torch.float, device=device))
        batch_loss.backward(retain_graph=True)
        optimizer.step()
        loss = batch_loss.item()
        total_loss += loss
        main_index = main_index + len(batch)
        loss = total_loss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
    dev_results = validate(valid_data)
    dev_file = open(
        "gmnbcbresult/" + args.graph_mode + "_dev_epoch_" + str(epoch + 1), mode="w"
    )
    for res in dev_results:
        dev_file.write(str(res) + "\n")
    dev_file.close()
    test_results = validate(test_data)
    res_file = open(
        "gmnbcbresult/" + args.graph_mode + "_epoch_" + str(epoch + 1), mode="w"
    )
    for res in test_results:
        res_file.write(str(res) + "\n")
    res_file.close()

    torch.save(model, "gmnmodels/gmnbcb" + str(epoch + 1))
