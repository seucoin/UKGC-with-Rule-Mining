import torch
from collections import defaultdict
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F


# 需要先将图转为id，然后加载到数据集，之后输入给模型，进行debug
class KnowledgeGraph:
    def __init__(self, data_path):
        with open(f"{data_path}/entities.txt") as e, open(
            f"{data_path}/relations.txt"
        ) as r:
            # 转id
            self.ents = [x.strip() for x in e.readlines()]
            self.rels = [x.strip() for x in r.readlines()]
            self.pos_rels = len(self.rels)
            self.rels += ["inv_" + x for x in self.rels] + ["<slf>"]
            self.e2id = {self.ents[i]: i for i in range(len(self.ents))}
            self.r2id = {self.rels[i]: i for i in range(len(self.rels))}
            self.id2r = {i: self.rels[i] for i in range(len(self.rels))}
            self.id2e = {i: self.ents[i] for i in range(len(self.ents))}

        # id四元组
        self.data = {}
        with open(f"{data_path}/train.txt") as f:
            train = [item.strip().split("\t") for item in f.readlines()]
            self.data["train"] = list( { (self.e2id[h], self.r2id[r], self.e2id[t], eval(s)) for h, r, t, s in train } )
        with open(f"{data_path}/test.txt") as f:
            test = [item.strip().split("\t") for item in f.readlines()]
            self.data["test"] = list( { (self.e2id[h], self.r2id[r], self.e2id[t], eval(s)) for h, r, t, s in test } )
        with open(f"{data_path}/valid.txt") as f:
            valid = [item.strip().split("\t") for item in f.readlines()]
            self.data["valid"] = list( { (self.e2id[h], self.r2id[r], self.e2id[t], eval(s)) for h, r, t, s in valid } )

        self.fact = defaultdict(dict)   # 二重字典，h,r双重键，值是（尾实体，置信度）元组构成的列表
        for h, r, t,s in self.data['train']:
            try:
                self.fact[h][r].add((t,s))
            except KeyError:
                self.fact[h][r] = set([(t,s)])
            try:
                self.fact[t][r+self.pos_rels].add((h,s))
            except KeyError:
                self.fact[t][r+self.pos_rels] = set([(h,s)])
        for h in self.fact:
            self.fact[h] = {r:list(ts) for r,ts in self.fact[h].items()}

        self.neighbors = defaultdict(dict)  # 二重字典，h，r双重键，值是尾实体列表呗
        for h, r, t,s in self.data['train']:
            try:
                self.neighbors[h][r].add(t)
            except KeyError:
                self.neighbors[h][r] = set([t])
            try:
                self.neighbors[t][r+self.pos_rels].add(h)
            except KeyError:
                self.neighbors[t][r+self.pos_rels] = set([h])
        for h in self.neighbors:
            self.neighbors[h] = {r:list(ts) for r,ts in self.neighbors[h].items()}


        # 获取稀疏矩阵
        sparse = data_path + "/sparse.pkl"
        try:
            with open(sparse, "rb") as db:
                self.relations = pickle.load(db)
        except:
            indices = [[] for _ in range(self.pos_rels)]  # train graph转换为稀疏矩阵，为什么只用正关系？
            values = [[] for _ in range(self.pos_rels)]
            no_repeat = defaultdict(set)
            for h, r, t, s in self.data["train"]:
                if (h, t) not in no_repeat[r]:
                    indices[r].append((h, t))  # 表示位置，即行列
                    values[r].append(s)  # 表示值，即行列位置上的元素，这里都是置信度
                    no_repeat[r].add((h, t))
            
            for i in range(self.pos_rels):
                if indices[i] == []:
                    indices[i].append((0,0))
                    values[i].append(0)

            indices = [torch.LongTensor(x).T for x in indices]
            values = [torch.FloatTensor(x) for x in values]
            size = torch.Size([len(self.ents), len(self.ents)])
            self.relations = [
                torch.sparse.FloatTensor(indices[i], values[i], size).coalesce()
                for i in range(self.pos_rels)
            ]
            # with open(f"{sparse}", "wb") as db:
            #     pickle.dump(self.relations, db)

        # 所有三元组，包括train，test，valid，以及rev三元组，(头实体,关系)的tuple作为键，尾实体集合作为值存储在filtered_dict字典中
        self.filtered_dict = defaultdict(set)
        triplets = self.data["train"] + self.data["valid"] + self.data["test"]
        for triplet in triplets:
            self.filtered_dict[(triplet[0], triplet[1])].add(triplet[2])
            self.filtered_dict[(triplet[2], triplet[1] + self.pos_rels)].add(triplet[0])


class MyDataset(Dataset):
    def __init__(self, kg, mode):
        self.kg = kg
        self.triples = kg.data[mode]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        h, r, t, s = self.triples[index]
        return h, r, t, s

    @staticmethod
    def collate_fn(data):
        query = torch.stack([torch.tensor([d[0], d[1]]) for d in data], dim=0)
        t = torch.stack([torch.tensor(d[2]) for d in data], dim=0)
        s = torch.stack([torch.tensor(d[3]) for d in data], dim=0)
        return query, t, s


if __name__ == "__main__":
    # 测试代码
    data_path = "/data/cyl/MyPaper/MyCodes/DATASETS/CN15k"
    kg = KnowledgeGraph(data_path)
    train_set = MyDataset(kg, "train")
    train_dataloader = DataLoader(
        train_set, batch_size=5, collate_fn=MyDataset.collate_fn
    )
    for batch in train_dataloader:
        print(batch)
        break
