import numpy as np
import pandas as pd
from tqdm import tqdm
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import log_loss, roc_auc_score, f1_score


class BKTModel():
    def __init__(self, n_iter=10, verbose=True):
        self.n_iter = n_iter
        self.n_s = 0
        self.verbose = verbose
        self.skills = []
        self.models = {}


    def preprocess(self, data):
        # Data sequencing (sorting)
        data = data.sort_values(by=['user_xid', 'skill_id', 'start_time'])
        grouped = data.groupby(by=['skill_id', 'user_xid'])

        curr_skill = 1
        skill_dict = {}
        value_dict = {"seq": [], "lengths": []}

        for value, group in tqdm(grouped, disable=not self.verbose):
            if value[0] != curr_skill:
                value_dict['seq'] = np.concatenate(value_dict['seq']).reshape(-1, 1)
                skill_dict[curr_skill] = value_dict
                value_dict = {"seq": [], "lengths": []}

            curr_skill = value[0]
            value_dict['seq'].append(group.discrete_score.to_numpy())
            value_dict['lengths'].append(group.shape[0])

        value_dict['seq'] = np.concatenate(value_dict['seq']).reshape(-1, 1).astype(int)
        skill_dict[curr_skill] = value_dict

        return skill_dict

    def fit(self, data):

        if self.verbose:
            print("Beginning data preprocessing.")
        skill_dict = self.preprocess(data)

        y_trues = []
        y_preds = []

        oov_prior = 0
        oov_transmat = np.zeros((2,2))
        oov_emmissionprob = np.zeros((2,2))

        disable = True
        if self.verbose:
            print("Finished data processing. Beginning fitting process.")
        for skill, data in tqdm(skill_dict.items(), disable=not self.verbose):
            self.models[skill] = CategoricalHMM(n_components=2, n_iter=self.n_iter, tol=1e-4)

            X = skill_dict[skill]['seq']
            lengths = skill_dict[skill]['lengths']

            if max(lengths) < 5 or len(np.unique(X)) != 2:
                continue

            self.models[skill].fit(X, lengths)
            self.skills.append(skill)

            oov_prior += self.models[skill].startprob_
            oov_transmat += self.models[skill].transmat_
            oov_emmissionprob += self.models[skill].emissionprob_

            state_probs = self.models[skill].predict_proba(X, lengths)
            y_pred = state_probs.dot(self.models[skill].emissionprob_[:,1])
            y_true = np.reshape(X,X.shape[0])

            y_trues.append(y_true)
            y_preds.append(y_pred)

        final_y_true = np.nan_to_num(np.concatenate(y_trues, axis=None))
        final_y_pred = np.nan_to_num(np.concatenate(y_preds, axis=None))

        ll = log_loss(final_y_true,final_y_pred)
        auc = roc_auc_score(final_y_true,final_y_pred)
        self.n_s = len(self.skills)

        if self.verbose:
            print("Finished model training. Printing final statistics...")
            print(f'Training Log Loss: {ll}')
            print(f'Training AUC: {auc}')

        prior = oov_prior / self.n_s
        trans = oov_transmat / self.n_s
        em = oov_emmissionprob / self.n_s

        oov_mod = CategoricalHMM(n_components=2)
        oov_mod.startprob_ = prior
        oov_mod.transmat_ = trans
        oov_mod.emissionprob_ = em

        self.models[-1] = oov_mod

        return auc


    def evaluate(self, data):
        if self.verbose:
            print("Beginning data preprocessing.")
        skill_dict = self.preprocess(data)
        y_trues = []
        y_preds = []

        for skill, data in tqdm(skill_dict.items(), disable=not self.verbose):
            X = skill_dict[skill]['seq']
            lengths = skill_dict[skill]['lengths']

            if not skill in self.skills:  # Handle OOV skills
                skill = -1

            state_probs = self.models[skill].predict_proba(X, lengths)
            y_pred = state_probs.dot(self.models[skill].emissionprob_[:, 1])
            y_true = np.reshape(X, X.shape[0])

            y_trues.append(y_true)
            y_preds.append(y_pred)

        final_y_true = np.nan_to_num(np.concatenate(y_trues, axis=None))
        final_y_pred = np.nan_to_num(np.concatenate(y_preds, axis=None))
        final_y_pred_classes = np.round(final_y_pred)

        ll = log_loss(final_y_true,final_y_pred)
        auc = roc_auc_score(final_y_true,final_y_pred)
        f1 = f1_score(final_y_true, final_y_pred_classes)

        if self.verbose:
            print(f'Eval Log Loss: {ll}')
            print(f'Eval AUC: {auc}')
            print(f'Eval F1: {f1}')

        return auc, ll, f1

    def get_num_params(self):
        return len(self.skills * 5)