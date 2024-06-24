"""
修改了新的解码方式，用邻接矩阵判断解码的关系是否相连，而不是用循环去判断
"""
""" This module will handle the text generation with beam search. """
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask
from collections import defaultdict
import numpy, time
from itertools import product
from datetime import datetime


class Translator(nn.Module):
    """Load a trained model and translate in beam search fashion."""

    def __init__(self, model, body_len, device=None, kg=None, decode=False):
        super(Translator, self).__init__()

        self.alpha = 0.7
        self.body_len = body_len
        
        self.n_rel_vocab = len(kg.rels)
        self.n_ent_vocab = len(kg.ents)
        # self.src_pad_idx = opt.src_pad_idx
        # self.n_trg_vocab = opt.trg_vocab_size
        # self.n_src_vocab = opt.src_vocab_size

        self.device = device

        self.model = model
        self.model.train()
        self.database = [x.to(device) for x in kg.relations]
        self.filtered_dict = kg.filtered_dict
        
        self.decode = decode
        if self.decode:
            self.graph = kg.neighbors
            self.decode_rule_num = 0
            self.the_rel = 0
            self.the_rel_min = 0
            self.the_all = 0
            self.id2r = kg.id2r
            self.id2e = kg.id2e
            self.rules = defaultdict(dict)
            self.decode_rule_num_filter = 0
            self.decode_file = 'cn15k_rules_0221.txt'

        self.register_buffer("thr", torch.FloatTensor([1e-20]))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = None
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        enc_output = None
        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(1)

        scores = torch.log(best_k_probs).view(self.batch_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx.squeeze()
        return enc_output, gen_seq, scores, dec_output

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(1)
        assert dec_output.size(1) == step

        gen_seq[:, step] = best_k2_idx.squeeze()

        return gen_seq, scores, dec_output

    def forwardAllNLP(self, query, t, s, attention, mode):
        num_pos_rel = self.n_rel_vocab // 2
        batch_size = query.size(0)
        database = [x.clone().detach() for x in self.database]
        for (hh, rr), tt in zip(query, t):
            if rr >= num_pos_rel:
                continue
            idxs = torch.where( (hh == database[rr].indices()[0]) & (tt == database[rr].indices()[1]) )
            database[rr].values()[idxs] = 0

        memories = F.one_hot(query[:,0], num_classes=self.n_ent_vocab).float().to_sparse()
        for step in range(self.body_len):
            added_results = torch.zeros(batch_size, self.n_ent_vocab).to(self.device)
            for r in range(num_pos_rel):
                for links, atta in zip( [database[r], database[r].transpose(0, 1)], [attention[:, step, r], attention[:, step, r + num_pos_rel]], ):
                    added_results = added_results + torch.sparse.mm( memories, links ).to_dense() * atta.unsqueeze(1)
            added_results = added_results + memories.to_dense() * attention[ :, step, -1 ].unsqueeze(1)
            # added_results = added_results / torch.max(self.thr, torch.sum(added_results, dim=1).unsqueeze(1))
            memories = added_results.to_sparse()
        memories = memories.to_dense()
        memories = memories / torch.max( self.thr, torch.sum(memories, dim=1).unsqueeze(1) )
        targets = F.one_hot(t, num_classes=self.n_ent_vocab).float()
        final_loss = -torch.sum( targets * torch.log(torch.max(self.thr, memories)), dim=1 )
        batch_loss = torch.mean(final_loss)
        if mode != "train":
            for i in range(batch_size):
                for idx in self.filtered_dict[(query[i,0].item(), query[i,1].item())]:
                    if idx != t[i].item():
                        memories[i, idx] = 0
        idxs = torch.argsort(memories, descending=True)
        indexs = torch.where(idxs == t.unsqueeze(-1))[1].tolist()
        return batch_loss, indexs

    def forward(self, query, t, s, mode="train"):
        batch_size = query.size(0)
        self.batch_size = batch_size
        # self.init_seq = trg[:, 1].unsqueeze(-1).clone().detach()
        self.init_seq = torch.zeros((batch_size, 1),dtype=torch.long).to(self.device)   # decoder输入
        self.blank_seqs = torch.zeros((batch_size,self.body_len+1),dtype=torch.long).detach().to(self.device)   # 存放解码序列，不参与梯度计算
        # src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        src_mask = None
        enc_output, gen_seq, scores, dec_output = self._get_init_state(query, src_mask)
        for step in range(2, self.body_len + 1):
            dec_output = self._model_decode( gen_seq[:, :step].clone().detach(), enc_output, src_mask )
            gen_seq, scores, dec_output = self._get_the_best_score_and_idx( gen_seq, dec_output, scores, step )

        gen_seq[:,0] = query[:,1]
        if self.decode:
            self.decode_rule(dec_output, query, t, s, mode)
            loss, index = 0, [0]
        else:
            loss, index = self.forwardAllNLP(query, t, s, dec_output, mode)

        return gen_seq, loss, index


    def decode_rule(self, attention, query, t, s, mode):
        def dfs(database,memories,rela_attention,step,body_len,tail,path):
            if step == body_len:
                return tail
            to_add = None
            step_attention = rela_attention[step]
            added_results = torch.zeros(1, self.n_ent_vocab).to(self.device)
            for r in range(self.n_rel_vocab):
                if r < self.n_rel_vocab//2:
                    if to_add == None:
                        to_add = torch.sparse.mm(memories,database[r]).to_dense()*step_attention[r]
                    else:
                        to_add = torch.cat([to_add,torch.sparse.mm(memories,database[r]).to_dense()*step_attention[r]])
                elif r < self.n_rel_vocab-1:
                    if to_add == None:
                        to_add = torch.sparse.mm(memories,database[r-self.n_rel_vocab//2].transpose(0,1)).to_dense()*step_attention[r]
                    else:
                        to_add = torch.cat([to_add,torch.sparse.mm(memories,database[r-self.n_rel_vocab//2].transpose(0,1)).to_dense()*step_attention[r]])
                else:
                    if to_add == None:
                        to_add = memories.to_dense()*step_attention[r]
                    else:
                        to_add = torch.cat([to_add,memories.to_dense()*step_attention[r]])
                added_results = added_results + torch.sum(to_add,dim=0)
            
            added_results = added_results.to_dense() / torch.max(self.thr, torch.sum(added_results, dim=1).unsqueeze(1))
            memories = added_results.to_sparse()

            target = dfs(database,memories,rela_attention,step+1,body_len,tail,path)
            path.append(target)
            
            rel_idx = torch.argmax(to_add[:,target])
            path.append(rel_idx)

            if step == 0:
                return

            if rel_idx < self.n_rel_vocab//2:
                adj_matrix = database[rel_idx].to_dense()
            elif rel_idx < self.n_rel_vocab-1:
                adj_matrix = database[rel_idx-self.n_rel_vocab//2].transpose(0,1).to_dense()
            else:
                adj_matrix = torch.eye(self.n_ent_vocab)

            adj_vec = adj_matrix[:,target].to(self.device)
            dots = memories*adj_vec
            return torch.argmax(dots.to_dense())



        num_pos_rel = self.n_rel_vocab // 2
        batch_size = query.size(0)
        database = [x.clone().detach() for x in self.database]
        
        for batch in range(batch_size):
            head_id = query[batch,0]
            rela_id = query[batch,1]
            tail_id = t[batch]
            conf = s[batch]

            rela_attention = attention[batch]

            idxs = torch.where((head_id == database[rela_id].indices()[0]) & (tail_id == database[rela_id].indices()[1]))
            database[rela_id].values()[idxs] = 0
            
            path = []
            memories = F.one_hot(torch.tensor([head_id]),num_classes=self.n_ent_vocab).float().to_sparse().to(self.device)
            dfs(database,memories,rela_attention,0,self.body_len,tail_id,path)
            path_item = list(reversed([x.item() for x in path]))
            path_item = [head_id.item()]+path_item

            for i in range(len(path_item)):
                if i%2 == 0:
                    path_item[i] = self.id2e[path_item[i]]
                else:
                    path_item[i] = self.id2r[path_item[i]]
            
            triple = [str(conf.item()),self.id2e[head_id.item()],self.id2r[rela_id.item()],self.id2e[tail_id.item()]]
            with open(self.decode_file,'a') as f:
                f.write(mode+'\t'+'\t'.join(triple+path_item)+'\n')

        

        # def decode_rule(self, dec_output, query, mode):
        #     relation_attention_list = dec_output
        #     batch_size = query.size(0)
        #     num_step = self.body_len
        #     for batch in range(batch_size):
        #         paths = {t + 1: [] for t in range(num_step)}
        #         # paths at hop 0, in the format of ([rel1,..],[ent1,..],weight)
        #         paths[0] = [([-1], [query[batch, 0].item()], 1.0)]
        #         relation_attentions = relation_attention_list[batch]
        #         for step in range(num_step):
        #             if not paths[step]:
        #                 break
        #             relation_attention_ori = relation_attentions[step]
        #             for rels, pths, wei in paths[step]:
        #                 if pths[-1] not in self.graph:
        #                     continue
        #                 # select relations(including self-loop) connected to the tail of each path
        #                 sel = torch.LongTensor(list(self.graph[pths[-1]].keys()) + [self.n_rel_vocab - 1])
        #                 relation_attention = torch.zeros(self.n_rel_vocab).to(self.device)
        #                 relation_attention[sel] = relation_attention_ori[sel].clone()
        #                 rel_att_max = torch.max(relation_attention).item()
        #                 relation_attention /= rel_att_max

        #                 for rr in torch.nonzero(relation_attention> max(self.the_rel, self.the_rel_min / rel_att_max)):  # relations which exceed threshold
        #                     # 归一化之后大于self.the_rel，归一化之前大于self.thr_rel_min
        #                     rr = rr.item()
        #                     if rr == self.n_rel_vocab - 1:  # <slf>
        #                         paths[step + 1].append((rels + [rr],pths + [pths[-1]],wei * relation_attention[rr].item(),))
        #                     elif rr in self.graph[pths[-1]].keys():
        #                         for tail in self.graph[pths[-1]][rr]:
        #                             paths[step + 1].append((rels + [rr],pths + [tail],wei * relation_attention[rr].item(),))

        #         for path in paths[step + 1]:
        #             rels, pths, wei = path
        #             if path[2] > self.the_all:
        #                 self.decode_rule_num += 1
        #                 print( "\rWrite {}-{} Rule(s)".format( self.decode_rule_num, self.decode_rule_num_filter ), end="", )
        #                 head_rule = self.id2r[query[batch, 1].item()]
        #                 rule_body = "^".join([self.id2r[r] for r in rels[1:]])
        #                 try:
        #                     self.rules[head_rule][rule_body].append(wei)
        #                 except KeyError:
        #                     self.rules[head_rule][rule_body] = [wei]
        #                     self.decode_rule_num_filter += 1
        #                     with open(self.decode_file, "a") as f:
        #                         f.write(mode+'\t'+self.id2e[query[batch,0].item()]+'\t'+head_rule + "<-" + rule_body + "\n")
