from __future__ import print_function

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from rdchiral.main import rdchiralRunText

from .mlp_policies import load_parallel_model, preprocess


def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)


class MLPModel(object):
    def __init__(self, state_path, template_path, device=-1, fp_dim=2048):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device >= 0:
            self.net.to(device)

    def run(self, x, topk=10):
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr, [-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        if self.device >= 0:
            arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds, dim=1)
        if self.device >= 0:
            preds = preds.cpu()
        probs, idx = torch.topk(preds, k=topk)
        # probs = F.softmax(probs,dim=1)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        templates = []
        for i, rule in enumerate(rule_k):
            out1 = []
            try:
                out1 = rdchiralRunText(rule, x)
                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                if len(out1) == 0:
                    continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item() / len(out1))
                    templates.append(rule)
            # out1 = rdchiralRunText(x, rule)
            except ValueError:
                pass
            except RuntimeError as e:
                print(f"RuntimeError encountered in rdchiralRunText: {e}")
                pass
        if len(reactants) == 0:
            return None
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if "." in r:
                str_list = sorted(r.strip().split("."))
                reactants_d[".".join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))

        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]
        return {"reactants": reactants, "scores": scores, "template": templates}
