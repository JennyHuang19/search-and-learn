import json
from tqdm import tqdm
import argparse
# pip install rouge_score
from rouge_score import rouge_scorer
# pip install codebelu
# pip install tree-sitter-python
# from codebleu import calc_codebleu
import numpy as np
import tokenize
import io

## DEBUG
import sys
import pdb
import traceback


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    pdb.pm()  # post-mortem debugger


sys.excepthook = debughook
## /DEBUG

# tokenize.generate_tokens(io.StringIO(code).readline)

# https://gist.github.ibm.com/ramon-astudillo/e89eba2ecf25c233bf197907df4f6fd0
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def read_data(score_files):
    # collect all samples for the same input
    pred_by_idx = dict()
    for score_file in score_files:
        for i, x in enumerate(open(score_file).readlines()):
            sample = json.loads(x)
            # not all formats have and index, if not assume its just the position
            if "idx" in sample:
                index = sample["idx"]
            else:
                index = i
            # start or update for this example
            if index in pred_by_idx:
                for k, v in sample.items():
                    pred_by_idx[index][k] += v
            else:
                pred_by_idx[index] = dict(sample)
        print("Collected {}: {}".format(score_file, len(sample["pred"])))

    indices, test_data = zip(*sorted(dict(pred_by_idx).items(), key=lambda x: x[0]))

    # assert len(set(indices)) == 500, "Indices repeated or missing, expected 500"
    return test_data 


def rouge_similarity(hypotheis, reference):
    return scorer.score(hypotheis, reference)["rougeL"].fmeasure


def codebleu_similarity(prediction, reference):
    result = calc_codebleu([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return result['codebleu']


def python_tokenize(string):
    tokens = []
    try:
        for y in tokenize.generate_tokens(io.StringIO(string).readline): 
            if y.string:
                tokens.append(y.string)
    except (tokenize.TokenError, IndentationError):
        return string

    except Exception as e:    
        from pdb import set_trace; set_trace() # noqa
        return string
        
    return tokens


def add_mbrd_scores(test_data, similarity_fun=None, tokenize=False):

    # select smilarity function
    if similarity_fun == None:
        similarity_fun = lambda x, y: x == y

    # loop over samples
    for samples in tqdm(test_data, desc="mbrd"):
        num_samples = len(samples["pred"])

        cache = dict()
        if tokenize:
            predictions = [" ".join(python_tokenize(x)) for x in samples["pred"]]
        else:
            predictions = samples["pred"]
    
        def similarity(pred, i, j):
            """
            similarity function
            """
            # cache (permutation invariant)
            key = tuple(sorted([pred[i], pred[j]]))
            if key in cache:
                pass
            elif len(set(key)) == 1:
                cache[key] = 1.0
            else:    
                cache[key] = similarity_fun(*key)
            return cache[key]

        # compute expectation approximation
        samples["mbrd_scores"] = [] 
        for i in range(num_samples):
            similarities = []
            for j in range(num_samples):
                similarities.append(similarity(predictions, i, j))
            samples["mbrd_scores"].append(similarities)
                
    return test_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score_files",
        nargs="+",
        required=True,
        help="predictions file containing the samples to score with prm",
    )
    parser.add_argument(
        "--out_score_files_mbrd",
        required=True,
        help="File containing the added MBRD scores",
    )
    return parser.parse_args()

def main():
    # python add_mbrd.py --score_files outputs/ibm-granite/granite-3.3-8b-it/numinamath500/phi4_v2-seed*/prm_scores.jsonl --out_score_files_mbrd outputs/ibm-granite/granite-3.3-8b-it/numinamath500/phi4_v2-rouge-mbrd.jsonl
    args = parse_args()

    # read inputs
    test_data = read_data(args.score_files)

    # add MBRD scores
    test_data = add_mbrd_scores(test_data, rouge_similarity, tokenize=False)

    # add MBRD scores
    with open(args.out_score_files_mbrd, "w") as fid:
        for line in test_data:
            fid.write(json.dumps(line) + "\n")
    print(args.out_score_files_mbrd)        

if __name__ == "__main__":
    main()
