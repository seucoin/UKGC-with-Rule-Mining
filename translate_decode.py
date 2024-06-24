""" Translate input text with trained model. """

import torch
import argparse, os, time

from transformer.Models import Transformer
from transformer.Translator import Translator
from transformer.dataset import MyDataset, KnowledgeGraph
from transformer.Optim import ScheduledOptim
import torch.optim as optim
import numpy as np
import random
import wandb
import os
from datetime import datetime


def hit_mrr(hits, starttime):
    return (
        "MRR:{:.5f} @1:{:.5f} @3:{:.5f} @10:{:.5f} LOS:{:.5f} Time:{:.1f}secs".format(
            hits[10] / hits[12],
            hits[0] / hits[12],
            hits[0:3].sum() / hits[12],
            hits[0:10].sum() / hits[12],
            hits[11] / hits[12],
            time.time() - starttime,
        )
    )


def run( translator, data_loader, mode, optimizer, device, epoch, logfile, starttime, decode ):
    pred_line = []
    hits = np.zeros(13)  # [0:10] for hit, [10] for mrr, [11] for loss, [12] for cnt
    for i, (query, t, s) in enumerate(data_loader):
        pred_seq, loss, indexL = translator( query.to(device), t.to(device), s.to(device), mode )
        if decode == True:
            continue
        print(f"\r {mode} {epoch}-{i}/{len(data_loader)}", end="    ")
        for j, index in enumerate(indexL):
            if index < 10:
                hits[index] += 1
            hits[10] += 1 / (index + 1)
            hits[12] += 1
        hits[11] += loss.item()


        if mode == 'train':
            wandb.log({f"{mode}_MRR": hits[10]/hits[12]})
            wandb.log({f"{mode}_loss": loss.item()})

        if mode == "train":
            loss.backward()
            optimizer.step_and_update_lr()
            wandb.log({"lr": optimizer.lr})
            optimizer.zero_grad()
        print(hit_mrr(hits, starttime), end="       ")

    if mode != 'train' and not decode:
        wandb.log({f"{mode}_MRR": hits[10]/hits[12]})
        wandb.log({f"{mode}_loss":hits[11]/hits[12]})
    if not decode:
        print(f"\r         {mode}-{epoch}  " + hit_mrr(hits, starttime) + "     ")
        with open(logfile, "a") as log:
            log.write(f"{mode}-{epoch}  " + hit_mrr(hits, starttime) + "\n")
    return pred_line


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    kg_path = "/data/cyl/MyPaper/No_sub_Ruleformer4UKG/DATASET/CN15k"
    kg = KnowledgeGraph(kg_path)
    train_data = MyDataset(kg, "train")
    valid_data = MyDataset(kg, "valid")
    test_data = MyDataset(kg, "test")

    batch_size = 512

    train_loader = torch.utils.data.DataLoader( dataset=train_data, batch_size=batch_size, shuffle=False, collate_fn=MyDataset.collate_fn, )  # , num_workers=16
    valid_loader = torch.utils.data.DataLoader( dataset=valid_data, batch_size=batch_size, shuffle=False, collate_fn=MyDataset.collate_fn, )
    test_loader = torch.utils.data.DataLoader( dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn=MyDataset.collate_fn, )

    device = torch.device("cpu")

    d_word_vec = 384
    d_model = 384
    d_inner = 1024
    n_layers = 4
    n_head = 6
    d_k = 64
    d_v = 64
    dropout = 0.1
    n_position = 200
    lr_mul = 0.02
    n_warmup_steps = 5000
    
    model = Transformer(
        n_ent_vocab=len(kg.ents),
        n_rel_vocab=len(kg.rels),
        d_word_vec=d_word_vec,
        d_model=d_model,
        d_inner=d_inner,
        n_layers=n_layers,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout,
        n_position=n_position,
    )

    decode_rule = True
    translator = Translator(model=model, body_len=3, device=device, kg=kg, decode = decode_rule).to(device)
    ckpt = '/data/cyl/MyPaper/No_sub_Ruleformer4UKG/EXPS/20240208_000822_CN15K_un/Translator40.ckpt'
    if ckpt and decode_rule:
        print("CKPT APPOINTED!!! DECODE MODE!!!")
        decode_rule = True
        translator.load_state_dict(torch.load(ckpt))

    optimizer = ScheduledOptim( optim.Adam(translator.parameters(), betas=(0.9, 0.98), eps=1e-09), lr_mul, d_model, n_warmup_steps, )
    
    starttime = time.time()

    num_epoch = 1
    
    savestep = 0

    if not decode_rule:
        wandb.init(
                # set the wandb project where this run will be logged
                project='MyModel',
                # track hyperparameters and run metadata
                config={
                    "batch_size": batch_size,
                    "epochs": num_epoch,
                },
            )

    logfile = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(f'EXPS/{logfile}_decode'):
        os.mkdir(f'EXPS/{logfile}_decode')
    

    config = {
        "data_path":kg_path,
        "batch_size":batch_size,
        "d_word_vec": d_word_vec,
        "d_model": d_model,
        "d_inner": d_inner,
        "n_layers": n_layers,
        "n_head": n_head,
        "d_k": d_k,
        "d_v": d_v,
        "dropout": dropout,
        "n_position": n_position,
        "lr_mul":lr_mul,
        "n_warmup_steps":n_warmup_steps,
        "num_epoch":num_epoch,
        "save_step":5,
        "decode_rule": decode_rule
    }

    # 指定要写入的文件路径
    file_path = f'EXPS/{logfile}_decode/config.txt'

    # 将参数写入文件
    with open(file_path, "w") as file:
        for key, value in config.items():
            file.write(f"{key} = {value}\n")
        

    for epoch in range(num_epoch):
        if not decode_rule:
            run( translator, train_loader, "train", optimizer, device, epoch + 1, logfile, starttime, decode_rule )
        if savestep and (epoch + 1) % savestep == 0:
            torch.save(translator.state_dict(), f"EXPS/{logfile}/Translator{epoch+1}.ckpt")
        if decode_rule or (epoch + 1) % 5 == 0:
            with torch.no_grad():
                if decode_rule:
                    run( translator, train_loader, "train", optimizer, device, epoch + 1, logfile, starttime, decode_rule )
                run( translator, valid_loader, "valid", optimizer, device, epoch + 1, logfile, starttime, decode_rule )
                run( translator, test_loader, "test", optimizer, device, epoch + 1, logfile, starttime, decode_rule )

    print("[Info] Finished.")


if __name__ == "__main__":
    main()
