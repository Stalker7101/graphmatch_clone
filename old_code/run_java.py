import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm, trange
from old_code.create_clone_java import create_ast, creategmndata, create_separate_graph
import models

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default="gcj")
parser.add_argument("--graphmode", default="astandnext")
parser.add_argument("--nextsib", default=False)
parser.add_argument("--ifedge", default=False)
parser.add_argument("--whileedge", default=False)
parser.add_argument("--foredge", default=False)
parser.add_argument("--blockedge", default=False)
parser.add_argument("--nexttoken", default=False)
parser.add_argument("--nextuse", default=False)
parser.add_argument("--data_setting", default="0")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()

device = torch.device("cuda:0")
# device=torch.device('cpu')
astdict, vocablen, vocabdict = create_ast()
print(astdict.keys())
treedict = create_separate_graph(
    astdict,
    vocablen,
    vocabdict,
    device,
    mode=args.graphmode,
    next_sib=args.nextsib,
    if_edge=args.ifedge,
    while_edge=args.whileedge,
    for_edge=args.foredge,
    block_edge=args.blockedge,
    next_token=args.nexttoken,
    next_use=args.nextuse,
)
traindata, validdata, testdata = creategmndata(
    args.data_setting, treedict, vocablen, vocabdict, device
)
num_layers = int(args.num_layers)
model = models.GMNnet(
    vocablen, embedding_dim=100, num_layers=num_layers, device=device
).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CosineEmbeddingLoss()
criterion2 = nn.MSELoss()


def create_batches(data):
    batches = [
        data[graph : graph + args.batch_size]
        for graph in range(0, len(data), args.batch_size)
    ]
    return batches


def test(dataset):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = []
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
        x1 = torch.tensor(x1, dtype=torch.long, device=device)
        x2 = torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1 != None:
            edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        prediction = model(data)
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
    p = 0.0
    r = 0.0
    f1 = 0.0
    if tp + fp == 0:
        print("precision is none")
        return
    p = tp / (tp + fp)
    if tp + fn == 0:
        print("recall is none")
        return
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print("precision")
    print(p)
    print("recall")
    print(r)
    print("F1")
    print(f1)
    return results


epochs = trange(args.num_epochs, leave=True, desc="Epoch")
for epoch in epochs:  # without batching
    print(epoch)
    batches = create_batches(traindata)
    totalloss = 0.0
    main_index = 0.0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
        optimizer.zero_grad()
        batchloss = 0
        for data, label in batch:
            label = torch.tensor(label, dtype=torch.float, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 != None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
            data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            prediction = model(data)
            cossim = F.cosine_similarity(prediction[0], prediction[1])
            batchloss = batchloss + criterion2(cossim, label)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss += loss
        main_index = main_index + len(batch)
        loss = totalloss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
    devresults = test(validdata)
    devfile = open(
        "gcjresult/" + args.graphmode + "_dev_epoch_" + str(epoch + 1), mode="w"
    )
    for res in devresults:
        devfile.write(str(res) + "\n")
    devfile.close()
    testresults = test(testdata)
    resfile = open("gcjresult/" + args.graphmode + "_epoch_" + str(epoch + 1), mode="w")
    for res in testresults:
        resfile.write(str(res) + "\n")
    resfile.close()

    # torch.save(model,'models/gmngcj'+str(epoch+1))
    # for start in range(0, len(traindata), args.batch_size):
    # batch = traindata[start:start+args.batch_size]
    # epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
