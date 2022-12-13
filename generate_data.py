from pathlib import Path
from hmmlearn.hmm import MultinomialHMM
from hmm_dist_est import ProbDist, ProbDistVit
import numpy as np
import random
from functools import partial
from string import ascii_lowercase
from itertools import permutations
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
import contextlib
from copy import deepcopy, copy
from joblib import Parallel, delayed
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
import json


def save_tokenizer_json(vocab, filename):
    vocab = list(vocab)
    vocab = ['[endoftext]'] + vocab
    vocab_dict = {v: i for i, v in enumerate(vocab)}
    tokenizer_config = {"version":"1.0","truncation": None,"padding": None,"added_tokens":[{"id":0,"special":True,"content":"[endoftext]","single_word":False,"lstrip":False,"rstrip":False,"normalized":False}],"normalizer":None,"pre_tokenizer":{"type":"Whitespace"},"post_processor":None,"decoder":None,"model":{"type":"WordLevel","vocab":vocab_dict,"unk_token":"<unk>"}}
    with open(filename, 'w') as f:
        json.dump(tokenizer_config, f)


def softmax(x, temp=1.0, axis=None):
    x /= temp
    if axis is None:
        x -= np.amax(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x -= np.expand_dims(np.amax(x, axis=axis), axis=axis)
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis=axis)


def generate_transmat_block(
        n_components, perm_samples=10, transition_temp=1.0):
    mixing = softmax(np.random.rand(perm_samples) - 0.5, transition_temp)
    mixing = mixing[:, np.newaxis, np.newaxis]
    perm_samples = [np.eye(n_components)[np.random.permutation(n_components)] for i in range(perm_samples)]
    transmat = np.sum(mixing * perm_samples, axis=0)

    return transmat


def generate_similar_transmat_blocks(
        n_components, perm_samples_num=10, transition_temp=1.0, num=3):
    transmat_list = []
    epsilon = 1e-4
    mixing = softmax(np.random.rand(perm_samples_num) - 0.5, transition_temp)
    perm_samples = [np.eye(n_components)[np.random.permutation(n_components)] for i in range(perm_samples_num)]
    transmat_list.append(np.sum(mixing[:, np.newaxis, np.newaxis] * perm_samples, axis=0))
    for i in range(num-1):
        mixing_random = copy(mixing)
        mixing_random[0] = mixing_random[0] + epsilon * random.uniform(0, 1)
        mixing_random = softmax(mixing_random, transition_temp)
        mixing_random = mixing_random[:, np.newaxis, np.newaxis]
        transmat_list.append(np.sum(mixing_random * perm_samples, axis=0))
    return transmat_list


def combine_transmats(mat1, mat2):
    # combine by tiling mat1 and scaling with mat2
    n = mat1.shape[0]
    m = mat2.shape[0]
    mat = np.zeros((m*n, m*n))
    for i in range(m):
        for j in range(m):
            mat[i*n: (i+1)*n, j*n: (j+1)*n] = mat1 * mat2[i,j]
    return mat


@contextlib.contextmanager
def local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def generate_hmm_parameters(
        n_values, n_slots, n_symbols, all_values, perm_samples=10, transition_temp=1.0,
        start_temp=1.0, value_transmat_id_coeff=0.8, value_transmat_seed=1112):
    n_components = n_values * n_slots

    # generate parameters for HMM
    startprob = softmax(np.random.rand(n_components) - 0.5, start_temp)

    slot_transmat = generate_transmat_block(
        n_slots, perm_samples=n_slots, transition_temp=transition_temp)

    if args.prior_values:
        value_transmat = generate_transmat_block(
            n_values, perm_samples=n_values, transition_temp=transition_temp)
        # bias the value transmat towards identity
        value_transmat = (1-value_transmat_id_coeff) * value_transmat + value_transmat_id_coeff * np.eye(n_values)
    else:
        with local_seed(value_transmat_seed):
            value_transmat = generate_transmat_block(
                n_values, perm_samples=n_values, transition_temp=transition_temp)
            # bias the value transmat towards identity
            value_transmat = (1-value_transmat_id_coeff) * value_transmat + value_transmat_id_coeff * np.eye(n_values)

    transmat = combine_transmats(slot_transmat, value_transmat)

    # this is actually the same for all hmms, given all_values
    epsilon = 1
    emissionprob = np.ones((n_components, n_symbols)) * (1 - epsilon) / n_symbols
    for i in range(n_components):
        # deterministic given slot and value vector
        slot_idx = i % n_slots
        value_idx = i // n_slots
        emissionprob[i, all_values[value_idx, slot_idx]] = epsilon

    return startprob, transmat, emissionprob, slot_transmat, value_transmat

def generate_similar_hmm_parameters(
        n_values, n_slots, n_symbols, all_values, transition_temp=1.0,
        start_temp=1.0, value_transmat_id_coeff=0.8, num_hmms=3):
    n_components = n_values * n_slots
    # generate parameters for HMM
    startprob = softmax(np.random.rand(n_components) - 0.5, start_temp)
    slot_transmats = generate_similar_transmat_blocks(
        n_slots, perm_samples_num=n_slots, transition_temp=transition_temp, num=num_hmms)

    value_transmat = generate_transmat_block(
        n_values, perm_samples=n_values, transition_temp=transition_temp)
    # bias the value transmat towards identity
    value_transmat = (1-value_transmat_id_coeff) * value_transmat + value_transmat_id_coeff * np.eye(n_values)

    transmats = [combine_transmats(slot_transmat, value_transmat) for slot_transmat in slot_transmats]

    # this is actually the same for all hmms, given all_values
    epsilon = 1
    emissionprob = np.ones((n_components, n_symbols)) * (1 - epsilon) / n_symbols
    for i in range(n_components):
        # deterministic given slot and value vector
        slot_idx = i % n_slots
        value_idx = i // n_slots
        emissionprob[i, all_values[value_idx, slot_idx]] = epsilon
    ret = []
    for i in range(num_hmms):
        ret.append((startprob, transmats[i], emissionprob, slot_transmats[i], value_transmat))
    return ret


def sample_from_hmm(hmm, length, seed=None):
    x, h = hmm.sample(n_samples=length, random_state=seed)
    return x.T[0], h


def get_default_sampler(hmm):
    return partial(sample_from_hmm, hmm=hmm)


def get_default_scorer(hmm):
    def score(x):
        proba = hmm.predict_proba([x])
        proba_last = proba[-1]
        proba_next_hidden = hmm.transmat_.T @ proba_last
        proba_next_emission = hmm.emissionprob_.T @ proba_next_hidden
        return proba_next_emission
    return score


def letter_generator(num):
    counter = 0
    for i in range(1, len(ascii_lowercase)):
        for perm in permutations(ascii_lowercase, i):
            yield ''.join(perm)
            counter += 1
            if counter >= num:
                return


def apply_vocab(tokens, vocab):
    return [vocab[tok] for tok in tokens]


def invert_vocab(tokens, vocab_to_int):
    return [vocab_to_int[tok] for tok in tokens]


def save_hmm_list(hmms, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(hmms, f)


def generate_hiddens_from_state(hmm, start_state, length):
    hiddens = [start_state]
    for i in range(length):
        hiddens.append(
            np.random.choice(hmm.transmat_.shape[1], p=hmm.transmat_[hiddens[-1], :]))
    return hiddens


def score(hmm, prompt, start_dist=None):
    if start_dist is not None:
        old_startprob = hmm.startprob_
        hmm.startprob_ = start_dist

    prompt = np.asarray(prompt).reshape(-1, 1)
    proba = hmm.predict_proba(prompt)
    proba_last = proba[-1]
    proba_next_hidden = hmm.transmat_.T @ proba_last
    proba_next_emission = hmm.emissionprob_.T @ proba_next_hidden

    if start_dist is not None:
        hmm.startprob_ = old_startprob

    return proba_next_emission


def estimate_pairwise_kl_divergence(hmm1, hmm2):
    prob_dist = ProbDist(hmm1, hmm2)
    y = []
    print("a")
    for i in range(35, 42):  # 45, 60
        lp_diff = 0
        times = 15  # 30
        for j in range(times):
            a, _, _ = prob_dist.dist(i)
            lp_diff += a
        y.append(lp_diff / times)
    return np.mean(y)

def plot_estimate_pairwise_kl_divergence(hmm1, hmm2):
    prob_dist = ProbDist(hmm1, hmm2)
    x = []
    y = []
    lp1 = []
    lp2 = []
    for i in range(2,100):
        x.append(i)
        a = 0
        b = 0
        c = 0
        times = 20
        for j in range(times):
            ap, bp, cp = prob_dist.dist(i)
            a += ap
            b += bp
            c += cp
        y.append(a / times)
        lp1.append(b / times)
        lp2.append(c / times)
    fig = plt.figure()
    plt.plot(x, y)
    plt.title('Distance vs # observationts', fontsize=14)
    plt.xlabel('number of observations, T', fontsize=14)
    plt.ylabel('distance', fontsize=14)
    fig.savefig('dist.png')
    fig2 = plt.figure()
    plt.plot(x, lp1, label = 'mu(Ot | lamda 0)')
    plt.plot(x, lp2, label = 'mu(Ot | lamba)')
    plt.xlabel('number of observations, T', fontsize=14)
    plt.ylabel('log probability', fontsize=14)
    plt.legend()
    fig2.savefig('log.png')
    return np.mean(y)


def generate_diverse_hmms(hmm_list, n_hmms):
    L = len(hmm_list)
    diff_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if i == j:
                diff_matrix[i][j] = -10
            else:
                diff_matrix[i][j] = estimate_pairwise_kl_divergence(hmm_list[i], hmm_list[j])
    used = {0}
    while len(used) < n_hmms:
        ind = -1
        max = -10
        for i in range(L):
            v = 0
            if i in used:
                continue
            for j in used:
                v = v + diff_matrix[i][j] + diff_matrix[j][i]
            if v > max:
                max = v
                ind = i
        used.add(ind)
    return used


def make_hmm_pred(prompt, hmms):
    # uniform prior over hmms, take average over prediction probs from each hmm
    # probs = Parallel(n_jobs=2)(delayed(score)(hmm) for hmm in hmms)
    probs = []
    for hmm in hmms:
        proba_next_emission = score(hmm, prompt)
        probs.append(proba_next_emission)
    avg_probs = np.mean(probs, axis=0)
    return np.argmax(avg_probs)


def generate_prompts(
        type_name, n_prompts, n_examples_per_prompts, num_slots, num_values,
        all_values, id_params, id_hmms, random_sample=False, hmms=None,
        prompt_length=2):

    prompts = []
    for n_examples_per_prompt in tqdm(n_examples_per_prompts):
        if (prompt_length + 1) * n_examples_per_prompt > 1024:
            continue
        for i in range(n_prompts):

            curr_hmm_idx = None
            if random_sample:
                start_slot = None
                assert(hmms is not None)
                curr_hmm_idx = np.random.choice(list(range(len(hmms))))
                curr_hmm = hmms[curr_hmm_idx]

                if type_name == 'ID_sample':
                    slots = []
                    values = []
                    # choose start such that we sample one start slot
                    start_slot = np.random.randint(low=1, high=num_slots)
                    for j in range(n_examples_per_prompt+1):
                        start_value = np.random.randint(low=0, high=num_values)
                        start_hidden_idx = start_value * num_slots + start_slot

                        h = generate_hiddens_from_state(curr_hmm, start_hidden_idx, length=prompt_length-1)
                        curr_slots = [curr_h % num_slots for curr_h in h]
                        curr_values = [curr_h // num_slots for curr_h in h]
                        slots += curr_slots
                        values += curr_values

                        # delimiter
                        slots += [0]
                        values += [values[-1]]
                elif type_name == 'OOD_sample':
                    # x y delim pattern
                    slots = []
                    values = []
                    for j in range(n_examples_per_prompt+1):
                        slot_pattern = list(np.random.randint(low=1, high=num_slots, size=prompt_length))
                        slots += slot_pattern
                        values += [np.random.randint(low=0, high=num_values)]*(prompt_length)

                        # delimiter
                        slots += [0]
                        values += [values[-1]]
                else:
                    raise ValueError("ID_sample or OOD_sample")

                prompt = [all_values[values[j], slots[j]] for j in range(len(slots))]

                # remove delimiter
                prompt = prompt[:-1]

                # test example

                test_value = values[-2]
                x = prompt[-prompt_length:-1]
                slot_pattern = slots[-(prompt_length + 1):]

                if start_slot is not None:
                    start_dist = np.zeros_like(curr_hmm.startprob_)
                    for c_idx in range(num_values):
                        start_dist[c_idx * num_slots + start_slot] = 1
                    start_dist /= start_dist.sum()
                else:
                    start_dist = np.ones_like(curr_hmm.startprob_) / curr_hmm.transmat_.shape[0]
                probs = score(curr_hmm, x, start_dist=start_dist)
                y = np.argmax(probs)

                # remove y
                prompt = prompt[:-1]

                prompt_ls = prompt

                prompt = ' '.join(apply_vocab(prompt, vocab))
                x = vocab[x]
                y = vocab[y]
            else:
                if type_name == 'ID':
                    slot_pattern = [np.random.randint(low=1, high=num_slots)]
                    idx = np.random.randint(low=0, high=len(id_params))
                    slot_transmat, value_transmat = id_params[idx]
                    for _ in range(prompt_length - 1):
                        slot_pattern.append(np.argmax(slot_transmat[slot_pattern[-1]]))
                    slot_pattern.append(0)
                elif type_name == 'OOD':
                    # x y delim pattern
                    slot_pattern = list(np.random.randint(low=1, high=num_slots, size=prompt_length))
                    slot_pattern += [0]
                slots = slot_pattern * (n_examples_per_prompt + 1)
                values = []
                for j in range(n_examples_per_prompt + 1):
                    values += [np.random.randint(low=0, high=num_values)]*len(slot_pattern)

                prompt = [all_values[values[j], slots[j]] for j in range(len(slots))]

                # remove delimiter
                prompt  = prompt[:-1]

                test_value = values[-1]
                x = prompt[-prompt_length:-1]
                y = prompt[-1]
                # remove y
                prompt = prompt[:-1]

                prompt_ls = prompt

                prompt = ' '.join(apply_vocab(prompt, vocab))
                x = vocab[x]
                y = vocab[y]

            res = {
                'text': prompt, 'label': y, 'x': x,
                'slot_pattern': slot_pattern,
                'test_value': test_value,
                'hmm_type': type_name, 'n_examples': n_examples_per_prompt,}
            if curr_hmm_idx is not None:
                res['hmm_id'] = curr_hmm_idx
            prompts.append(res)
    return prompts


def save_as_json(samples, save_path):
    df = pd.DataFrame(samples)
    df.to_json(save_path, orient='records', lines=True)


def generate_samples(num_samples, id_hmms, sample_length, random_data=False):
    id_samples = []
    for i in tqdm(range(num_samples)):
        j = np.random.choice(len(id_hmms))
        if not random_data:
            x, h = sample_from_hmm(id_hmms[j], sample_length)
        else:
            x = np.random.randint(low=0, high=len(vocab), size=sample_length)
            h = np.random.randint(low=0, high=n_components, size=sample_length)
        x = apply_vocab(x, vocab)
        id_samples.append({'text': ' '.join(x), 'hmm_idx': j, 'hmm_type': 'ID', 'hiddens': h})
    return id_samples


def samples_to_raw(samples, out_path):
    with open(out_path, 'w') as f:
        for sample in samples:
            f.write(sample['text'] + ' / ')


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate pretraining data for GINC')
    parser.add_argument('--start_temp', type=float, default=10.0, help="temperature on distribution of start hidden states")
    parser.add_argument('--transition_temp', type=float, default=0.1, help="temperature on transition probability matrices")
    parser.add_argument('--skip_resample', action='store_true', help="whether to re-sample the dataset")
    parser.add_argument('--n_symbols', type=int, default=10, help="number of symbols")
    parser.add_argument('--n_values', type=int, default=5, help="number of entities (values)")
    parser.add_argument('--n_slots', type=int, default=5, help="number of properties (slots)")
    parser.add_argument('--value_identity_coeff', type=float, default=0.8, help="mixing coefficient for identity matrix on value transition matrix")
    parser.add_argument('--random_data', action='store_true', help="whether to generate random data (not HMM)")
    parser.add_argument('--no_prior_slots', action='store_true', help="only one slot transition matrix instead of a family")
    parser.add_argument('--prior_values', action='store_true', help="use prior on values (many value transition matrices)")
    parser.add_argument('--n_train_samples', type=int, default=1000, help="number of training documents")
    parser.add_argument('--prompt_length', type=int, default=None, help="length of [x,y] sequence in prompt examples")
    parser.add_argument('--n_hmms', type=int, default=10, help="number of total hmms in the mixture")
    parser.add_argument('--list_of_n_hmms', type=str, help="generate a list of data files with the latter containing concepts from the prior. List must be ascending order and separated by comma. e.g. 1,3,5,6")
    parser.add_argument('--skip_all_generation', action='store_true', help="whether to skip everything")
    parser.add_argument('--root', type=str, help="root dir")
    parser.add_argument('--sample_similar_hmm', action='store_true')
    parser.add_argument('--sample_diver_hmm', action='store_true')
    args = parser.parse_args()

    if not args.skip_all_generation:

        dataset_id = f'GINC_trans{args.transition_temp}_start{args.start_temp}_nsymbols{args.n_symbols}_nvalues{args.n_values}_nslots{args.n_slots}'
        if args.value_identity_coeff != 0.8:
            dataset_id += f'_vic{args.value_identity_coeff}'

        if args.random_data:
            dataset_id += '_randomdata'
        if args.no_prior_slots:
            dataset_id += '_noslotprior'
        if args.prior_values:
            dataset_id += '_valueprior'

        if args.n_train_samples != 1000:
            dataset_id += f'_nsamples{args.n_train_samples}'

        if args.sample_similar_hmm:
            list_of_n_hmms = [args.n_hmms//2]
            dataset_id_list = [dataset_id + f'_nhmms{args.n_hmms}_least_diverse']
        elif args.sample_diver_hmm:
            list_of_n_hmms = [args.n_hmms]
            dataset_id_list = [dataset_id + f'_nhmms{args.n_hmms}_most_diverse']
        else:
            list_of_n_hmms = [int(x) for x in args.list_of_n_hmms.split(",")]
            if not list_of_n_hmms:
                list_of_n_hmms = [args.n_hmms]
            dataset_id_list = [dataset_id + f'_nhmms{i}' for i in list_of_n_hmms]

        seed = 1111
        np.random.seed(seed)
        random.seed(seed+2)
        n_symbols = args.n_symbols
        n_values = args.n_values
        n_slots = args.n_slots
        n_components = n_values * n_slots
        n_perm_samples = n_components
        num_val_samples = 100
        num_train_samples = args.n_train_samples
        sample_length = 10240
        val_sample_length = 1024
        n_prompts = 2500
        n_examples_per_prompts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        vocab = list(letter_generator(n_symbols))
        # replace delimiters with more interpretable tokens
        vocab = ['/'] + vocab[:-1]
        vocab = np.asarray(vocab)
        vocab_to_int = {k: i for i, k in enumerate(vocab)}

        # num_values number of num_slots sized lists of vocab words
        all_values = np.random.randint(low=1, high=len(vocab), size=(n_values, n_slots))
        # make sure every row has a delimiter
        all_values[:, 0] = 0

        hmm_list = []
        hmm_params = []
        if args.sample_similar_hmm:
            list_params = generate_similar_hmm_parameters(
                n_values,
                n_slots,
                n_symbols,
                all_values,
                transition_temp=args.transition_temp,
                start_temp=args.start_temp,
                value_transmat_id_coeff=args.value_identity_coeff,
                num_hmms=args.n_hmms//2)
            for t in list_params:
                startprob, transmat, emissionprob, slot_transmat, value_transmat = t
                hmm = MultinomialHMM(n_components=n_components)
                hmm.startprob_ = startprob
                hmm.transmat_ = transmat
                hmm.emissionprob_ = emissionprob
                hmm_list.append(hmm)
                hmm_params.append((slot_transmat, value_transmat))
        else:
            for i in range(list_of_n_hmms[-1]):
                startprob, transmat, emissionprob, slot_transmat, value_transmat = generate_hmm_parameters(
                    n_values,
                    n_slots,
                    n_symbols,
                    all_values,
                    perm_samples=n_perm_samples,
                    transition_temp=args.transition_temp,
                    start_temp=args.start_temp,
                    value_transmat_id_coeff=args.value_identity_coeff,
                    value_transmat_seed=seed+3)
                hmm = MultinomialHMM(n_components=n_components)
                hmm.startprob_ = startprob
                hmm.transmat_ = transmat
                hmm.emissionprob_ = emissionprob
                hmm_list.append(hmm)
                hmm_params.append((slot_transmat, value_transmat))
        for num_hmm, dataset_id in zip(list_of_n_hmms, dataset_id_list):
            data_dir = Path(args.root) / 'data'
            save_dir = data_dir / dataset_id
            save_dir.mkdir(exist_ok=True, parents=True)
            hmm_list_i = hmm_list[:num_hmm]
            hmm_params_i = hmm_params[:num_hmm]
            if not args.skip_resample:
                if args.sample_similar_hmm:
                    id_hmms = hmm_list_i
                    id_params = hmm_params_i
                    ood_hmms = []
                    ood_params = []
                    for i in range(list_of_n_hmms[-1]):
                        startprob, transmat, emissionprob, slot_transmat, value_transmat = generate_hmm_parameters(
                            n_values,
                            n_slots,
                            n_symbols,
                            all_values,
                            perm_samples=n_perm_samples,
                            transition_temp=args.transition_temp,
                            start_temp=args.start_temp,
                            value_transmat_id_coeff=args.value_identity_coeff,
                            value_transmat_seed=seed+3)
                        hmm = MultinomialHMM(n_components=n_components)
                        hmm.startprob_ = startprob
                        hmm.transmat_ = transmat
                        hmm.emissionprob_ = emissionprob
                        ood_hmms.append(hmm)
                        ood_params.append((slot_transmat, value_transmat))
                elif args.sample_diver_hmm:
                    # id_hmms_index = generate_diverse_hmms(hmm_list_i, args.n_hmms//2)
                    id_hmms_index = generate_diverse_hmms(hmm_list_i, 5)
                    id_hmms = [hmm_list_i[i] for i in id_hmms_index]
                    id_params = [hmm_params_i[i] for i in id_hmms_index]
                    ood_hmms = []
                    ood_params = []
                    for i in range(5):
                        startprob, transmat, emissionprob, slot_transmat, value_transmat = generate_hmm_parameters(
                            n_values,
                            n_slots,
                            n_symbols,
                            all_values,
                            perm_samples=n_perm_samples,
                            transition_temp=args.transition_temp,
                            start_temp=args.start_temp,
                            value_transmat_id_coeff=args.value_identity_coeff,
                            value_transmat_seed=seed+3)
                        hmm = MultinomialHMM(n_components=n_components)
                        hmm.startprob_ = startprob
                        hmm.transmat_ = transmat
                        hmm.emissionprob_ = emissionprob
                        ood_hmms.append(hmm)
                        ood_params.append((slot_transmat, value_transmat))
                else:
                    n_id_hmms = num_hmm // 2
                    id_hmms = hmm_list_i[:num_hmm][:n_id_hmms]
                    ood_hmms = hmm_list_i[n_id_hmms:]
                    id_params = hmm_params_i[:n_id_hmms]
                    ood_params = hmm_params_i[n_id_hmms:]
                breakpoint()
                print("Generating samples")
                if not args.no_prior_slots:
                    id_samples = generate_samples(
                        num_train_samples, id_hmms,
                        sample_length=sample_length,
                        random_data=args.random_data)
                    id_samples_val = generate_samples(
                        num_val_samples, id_hmms,
                        sample_length=val_sample_length,
                        random_data=args.random_data)
                else:
                    id_samples = generate_samples(
                        num_train_samples, [id_hmms[0]],
                        sample_length=sample_length,
                        random_data=args.random_data)
                    id_samples_val = generate_samples(
                        num_val_samples, [id_hmms[0]],
                        sample_length=val_sample_length,
                        random_data=args.random_data)
                # save the hmm parameters for later verification
                save_hmm_list(id_hmms, save_dir / 'id_hmms.pkl')
                save_hmm_list(ood_hmms, save_dir / 'ood_hmms.pkl')
                save_hmm_list(id_params, save_dir / 'id_params.pkl')
                save_hmm_list(ood_params, save_dir / 'ood_params.pkl')
                with open(save_dir / 'fixed_params.pkl', 'wb') as f:
                    pickle.dump(all_values, f)

                save_as_json(id_samples, save_dir / 'train.json')
                save_as_json(id_samples_val, save_dir / 'val.json')
                samples_to_raw(id_samples, save_dir / 'train.txt')
                samples_to_raw(id_samples_val, save_dir / 'val.txt')
            else:
                id_hmms = load(save_dir / 'id_hmms.pkl')
                ood_hmms = load(save_dir / 'ood_hmms.pkl')
                id_params = load(save_dir / 'id_params.pkl')
                ood_params = load(save_dir / 'ood_params.pkl')
                all_values = load(save_dir / 'fixed_params.pkl')

            print("Generate Tokenizer")

            tokenizer_path = save_dir / 'tokenizer.json'
            save_tokenizer_json(vocab, tokenizer_path)

            seed = 1114
            np.random.seed(seed)
            random.seed(seed+2)

            if not args.prompt_length:
                prompt_lengths = [3, 5, 8, 10]
            else:
                prompt_lengths = [args.prompt_length]

            for prompt_length in prompt_lengths:
                id_prompts = generate_prompts(
                    'ID_sample', n_prompts, n_examples_per_prompts, n_slots,
                    n_values, all_values, id_params, random_sample=True,
                    hmms=id_hmms, prompt_length=prompt_length, id_hmms=id_hmms)
                save_as_json(id_prompts, save_dir / f'id_prompts_randomsample_{prompt_length}.json')
                samples_to_raw(id_prompts, save_dir / f'id_prompts_randomsample_{prompt_length}.txt')

                ood_prompts = generate_prompts(
                    'ID_sample', n_prompts, n_examples_per_prompts, n_slots,
                    n_values, all_values, None, random_sample=True,
                    hmms=ood_hmms, prompt_length=prompt_length, id_hmms=id_hmms)
                save_as_json(ood_prompts, save_dir / f'ood_prompts_randomsample_{prompt_length}.json')
                samples_to_raw(ood_prompts, save_dir / f'ood_prompts_randomsample_{prompt_length}.txt')