import re
import torch
from torch import nn
import os
from .ELMoForManyLangs import elmo
from .postprocessing import _run_word_segmentation_with_dictionary, construct_dictionary
import numpy as np
import math
import json

def sort_list(li, piv=2,unsort_ind=None):
    ind = []
    if unsort_ind == None:
        ind = sorted(range(len(li[piv])), key=(lambda k: li[piv][k]))
    else:
        ind = unsort_ind
    for i in range(len(li)):
        li[i] = [li[i][j] for j in ind]
    return li, ind

def sort_numpy(li, piv=2,unsort=False):
    ind = np.argsort(-li[piv] if not unsort else li[piv], axis=0)
    for i in range(len(li)):
        if type(li[i]).__module__ == np.__name__ or type(li[i]).__module__ == torch.__name__:
            li[i] = li[i][ind]
        else:
            li[i] = [li[i][j] for j in ind]
    return li, ind

def sort_torch(li, piv=2,unsort=False):
    li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    for i in range(len(li)):
        if i == piv:
            continue
        else:
            li[i] = li[i][ind]
    return li, ind

def sort_by(li, piv=2, unsort=False):
    if type(li[piv]).__module__ == np.__name__:
        return sort_numpy(li, piv, unsort)
    elif type(li[piv]).__module__ == torch.__name__:
        return sort_torch(li, piv, unsort)
    else:
        return sort_list(li, piv, unsort)

class W2V_Embedder():
    def __init__(self, seq_len=0):
        self.syn0 = np.load(os.path.join(os.path.abspath(os.path.join(__file__ ,"..")), "CharEmb/word2vec_weights.npy"))
        self.word2idx = json.load(open(os.path.join(os.path.abspath(os.path.join(__file__ ,"..")), "CharEmb/word2idx.json")))
        self.seq_len = seq_len

    def __call__(self, sents, max_len=0, with_bos_eos=True, layer=-1, pad_matters=False):
        sents = [[self.sub_unk(x) for x in sent] for sent in sents]
        if with_bos_eos:
            sents = [["<bos>"]+x+["<eos>"] for x in sents]
        seq_lens = np.array([len(x) for x in sents], dtype=np.int64)
        if max_len != 0:
            pass
        elif self.seq_len != 0:
            max_len = self.seq_len
            seq_lens = np.array([min([len(x), max_len]) for x in sents], dtype=np.int64)
        else:
            max_len = seq_lens.max()
        
        embedded = np.zeros((len(sents), max_len, self.syn0.shape[1]), dtype=np.float32)
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                if sents[i][j] in self.word2idx:
                    embedded[i, j] = self.syn0[int(self.word2idx[sents[i][j]])]
                else:
                    pass
        return embedded, seq_lens

    def sub_unk(self, e):
        e = e.replace('，',',')
        e = e.replace('：',':')
        e = e.replace('；',';')
        e = e.replace('？','?')
        e = e.replace('！', '!')
        return e

class Embedder():
    def __init__(self, seq_len=0, use_cuda=True, device=None):
        if device == "cpu":
            use_cuda=False
        self.embedder = elmo.Embedder(model_dir="new.model", batch_size=512, use_cuda=use_cuda)
        self.seq_len = seq_len
        self.device = device
        if self.device != None:
            self.embedder.model.to(self.device)
        self.bos_vec, self.eos_vec, self.pad, self.oov = self.embedder.sents2elmo([["<bos>","<eos>","<pad>","<oov>"]], output_layer=0)[0]

    def __call__(self, sents, max_len=0, with_bos_eos=True, layer=-1, pad_matters=False):
        seq_lens = np.array([len(x) for x in sents], dtype=np.int64)
        sents = [[self.sub_unk(x) for x in sent] for sent in sents]
        if max_len != 0:
            pass
        elif self.seq_len != 0:
            max_len = self.seq_len
        else:
            max_len = seq_lens.max()
        emb_list = self.embedder.sents2elmo(sents, output_layer=layer)
        if not with_bos_eos:
            for i in range(len(emb_list)):
                if max_len - seq_lens[i] > 0:
                    if pad_matters:
                        emb_list[i] = np.concatenate([emb_list[i], np.tile(self.pad,[max_len - seq_lens[i],1])], axis=0)
                    else:
                        emb_list[i] = np.concatenate([emb_list[i], np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))])
                else:
                    emb_list[i] = emb_list[i][:max_len]
        elif with_bos_eos:
            for i in range(len(emb_list)):
                if max_len - seq_lens[i] > 0:
                    if pad_matters:
                        emb_list[i] = np.concatenate([
                            self.bos_vec[np.newaxis],
                            emb_list[i],
                            self.eos_vec[np.newaxis],
                            np.tile(self.pad, [max_len - seq_lens[i], 1])], axis=0)
                    else:
                        emb_list[i] = np.concatenate([
                            self.bos_vec[np.newaxis],
                            emb_list[i],
                            self.eos_vec[np.newaxis],
                            np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))], axis=0)
                else:
                    emb_list[i] = np.concatenate([self.bos_vec[np.newaxis], emb_list[i][:max_len],self.eos_vec[np.newaxis]], axis=0)
        embedded = np.array(emb_list, dtype=np.float32)
        seq_lens = seq_lens+2 if with_bos_eos else seq_lens
        return embedded, seq_lens

    def sub_unk(self, e):
        e = e.replace('，',',')
        e = e.replace('：',':')
        e = e.replace('；',';')
        e = e.replace('？','?')
        e = e.replace('！', '!')
        return e

class Utils():
    def __init__(self, w2v=False, elmo_device=None):
        self.elmo = Embedder(device=elmo_device) if not w2v else W2V_Embedder()
        self.ch_gex = re.compile(r'[\u4e00-\u9fff]+')
        self.eng_gex = re.compile(r'[a-zA-Z0-9０１２３４５６７８９\s]+')

    def string2list(self, line):
        ret = []
        temp_str = []
        for char in line:
            if self.eng_gex.findall(char).__len__() == 0:
                if temp_str.__len__() > 0:
                    stript = "".join(temp_str).strip()
                    if stript.__len__() > 0:
                        ret.append(stript)
                    temp_str = []
                ret.append(char)
            else:
                temp_str.append(char)
        if temp_str.__len__() > 0:
            ret.append("".join(temp_str).strip())
        return ret

    def f2h(self, s):
        # return re.sub(r"( |　)+", " ", s).strip()
        s = list(s)
        for i in range(len(s)):
            num = ord(s[i])
            if num == 0x3000:
                num = 32
            elif 0xFF01 <= num <= 0xFF5E:
                num -= 0xfee0
            s[i] = chr(num).translate(str.maketrans('﹕﹐﹑。﹔﹖﹗﹘　', ':,、。;?!- '))
        return re.sub(r"( |　)+", " ", "".join(s)).strip()

class Wordseg(nn.Module):
    def __init__(self,
        batch_size=64,
        embedding='w2v',
        device="cpu",
        elmo_use_cuda=True,
        mode="TW"):
        super(Wordseg, self).__init__()
        elmo_device="cpu" if not elmo_use_cuda else "cuda:0"
        hidden_size=300
        input_size=1024
        n_layers=3
        dropout = 0.33
        model_path = ".ckpt"
        if mode not in ["TW", "HK", "CN", "CN_PKU", "CN_MSR"]:
            raise Exception('mode should be among "TW", "HK", "CN", "CN_PKU" and "CN_MSR".')
        if embedding not in ["elmo", "w2v"]:
            raise Exception('embedding should be "elmo" or "w2v".')
        model_path = mode + model_path if mode != "CN" else "CN_PKU" + model_path
        model_path = os.path.join("elmo",model_path) if (embedding=="elmo") else os.path.join("w2v", model_path)
        model_path = os.path.join("models",model_path)
        model_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"..")), model_path)

        self.utils = Utils(w2v=(embedding=="w2v"),elmo_device=elmo_device)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,
                          batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, 2)
        self.to(self.device)
        self.load_model(filename=model_path)
        self.eval()

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = torch.from_numpy(input_seq).to(self.device)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        pred_prob = self.fc1(outputs)#nn.Softmax(dim=-1)(self.fc1(outputs))
        embedded.cpu()
        return pred_prob

    def cut(self, sents, merge_dict=None, force_dict=None):
        empty_pos = []
        valid_sents = []
        for i in range(len(sents)):
            if len(sents[i])==0:
                empty_pos.append(i)
            else:
                valid_sents.append(self.utils.string2list(self.utils.f2h(sents[i])))
        if valid_sents == []:
            return [[] for _ in empty_pos]
        ret = self.test(valid_sents, unsort=True)
        new_ret = []
        for sent in ret:
            new_ret.append([])
            for word in sent:
                if ' ' in word:
                    new_ret[-1] += word.split(' ')
                else:
                    new_ret[-1].append(word)
        if empty_pos != []:
            for i in empty_pos:
                new_ret.insert(i, [])
        if merge_dict is not None or force_dict is not None:
            refine_ret = []
            for sent in new_ret:
                if len(sent) == 0:
                    refine_ret.append([])
                else:
                    refine_ret.append(_run_word_segmentation_with_dictionary(sent, recommend_dictionary=merge_dict, coerce_dictionary=force_dict))
            return refine_ret
        else:
            return new_ret

    def test(self, sents, unsort=False):
        embedded, seq_lens = self.utils.elmo(sents)
        (embedded, seq_lens, sents), ind = sort_by([embedded, seq_lens, sents], piv=1)
        pred = self.forward(embedded, seq_lens)
        ans = torch.argmax(pred, dim=-1).cpu().numpy()[:, 1:]
        new_sents = []
        for i, sent in enumerate(sents):
            new_sent = []
            cur_word = []
            for j, char in enumerate(sent):
                if ans[i, j] == 0:
                    cur_word.append(char)
                else:
                    cur_word.append(char)
                    new_sent.append("".join(cur_word))
                    cur_word = []
            new_sents.append(new_sent)
        if unsort:
            (new_sents, ind), _ind = sort_by([new_sents, ind], piv=1, unsort=True)
        return new_sents

    def load_model(self, filename='model.ckpt', device=None):
        if device == None:
            device = self.device
        self.load_state_dict(torch.load(filename, map_location=device))

if __name__ == "__main__":
    seg = Wordseg(device="cuda:1", mode="TW")
    print(seg.cut(["今天天氣真好啊!", "潮水退了就知道，誰沒穿褲子。"]))
