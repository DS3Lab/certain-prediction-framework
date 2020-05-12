#!/usr/bin/env python3

import argparse
import collections
import csv
import functools
import itertools
import math
import re
import sys


KERNELS = {
    "euclidean" : lambda x, y: math.sqrt(sum((xx * yy) ** 2 for xx, yy in zip(x, y))),
    "manhattan" : lambda x, y: sum(abs(xx - yy) for xx, yy in zip(x, y)),
    "chebyshev" : lambda x, y: max(abs(xx - yy) for xx, yy in zip(x, y))
}

DEFAULT_KERNEL = "euclidean"
DEFAULT_K = 1

def compute_similarities(dataset, example, kernel=DEFAULT_KERNEL):
    if isinstance(kernel, str):
        kernel = KERNELS[kernel]

    distances = []
    for row in dataset:
        # The row candidates are simply a cartesian product of all cell candidates.
        row_candidates = itertools.product(*row)
        # Use the kernel to compute distances between each row candidate and the test example.
        distances.append([kernel(x, example) for x in row_candidates])

    # We represent similarities simply as 1/distance.
    similarities = [[1 / (x + sys.float_info.epsilon) for x in row] for row in distances]

    return similarities


def compute_probabilities(dataset_probs):
    probabilities = []
    for row in dataset_probs:
        row_probabilities = []
        for candidate_probs in itertools.product(*row):
            row_probabilities.append(functools.reduce((lambda x, y: x * y), candidate_probs))
        probabilities.append(row_probabilities)
    return probabilities


def q1(similarities, labels, k=DEFAULT_K):
    
    # Initialize helper variables.
    classes = sorted(list(set(labels)))
    answer = {}

    # Try to construct an bounday world for every class c, and for each one try to predict class c.
    for c in classes:

        # Construct best possible world for predicting class c.
        world = [max(s) if l == c else min(s) for (s, l) in zip(similarities, labels)]

        # Compute answer given by KNN for that world.
        top_k_set = sorted(zip(world, labels), reverse=True, key=lambda x: x[0])[:k]
        top_k_labels = [x[1] for x in top_k_set]
        counter = collections.Counter(top_k_labels)
        # Select winning class. Ties are broken by selecting the class that comes earlier in the sort order.
        winner = sorted(counter.most_common(), reverse=True, key=lambda x: (x[1], x[0]))[0][0]

        # Check if class c can be predicted.
        answer[c] = (winner == c)

    # Adjust answer to represent the definition of Q1: Can a class be certainly predicted?
    if sum([int(x) for x in answer.values()]) > 1:
        for c in answer:
            answer[c] = False

    return answer


def compute_dp(labels, alpha, i, m, c, k):

    dp = [[1] + [0 for _ in range(k)]]

    for n in range(len(alpha)):
        if labels[n] == c:
            dp.append([])
            for j in range(k + 1):
                if i == n:
                    if j > 0:
                        dp[-1].append(dp[-2][j-1])
                    else:
                        dp[-1].append(0)
                else:
                    if j > 0:
                        dp[-1].append(alpha[n] * dp[-2][j] + (m[n] - alpha[n]) * dp[-2][j-1])
                    else:
                        dp[-1].append(alpha[n] * dp[-2][j])

    return dp[-1]


def q2(similarities, labels, k=DEFAULT_K, probabilities=None):

    # Initialize helper variables.
    classes = sorted(list(set(labels)))
    answer = dict((c, 0) for c in classes)
    m = [len(row) for row in similarities]
    alpha = [0 for _ in range(len(similarities))]

    # If probabilities are provided, we are in probability mode and need to
    # assert that a probability distribution is defined over all row candidates.
    prob_mode = probabilities is not None
    if prob_mode:
        assert(all(len(similarities[i]) == len(probabilities[i]) for i in range(len(similarities))))
        m = [1.0 for _ in similarities]

    # Compute the sorting order of similarity candidates.
    similarities = [[(i, j, s) for (j, s) in enumerate(row)] for (i, row) in enumerate(similarities)]
    sorted_similarities = sorted(itertools.chain.from_iterable(similarities), key=lambda x: x[2])

    # Visit all similarity candidates in sorted order.
    for i, j, _ in sorted_similarities:

        # Maintain the similarity tally.
        if prob_mode:
            alpha[i] += probabilities[i][j]
        else:
            alpha[i] += 1

        # Compute the dynamic program for all classes.
        dp = {}
        for c in classes:
            dp[c] = compute_dp(labels, alpha, i, m, c, k)

        # Iterate over all possible label tally vectors.
        for gamma in itertools.filterfalse(lambda x: sum(x) != k, itertools.product(range(k+1), repeat=len(classes))):

            # Determine the winning label of this label tally vector.
            winner, winner_gamma = max(zip(classes, gamma), key=lambda x: x[1])

            # Compute label tally support as a product of label supports obtained from the dynamic program.
            support = 1
            for idx, c in enumerate(classes):
                support *= dp[c][gamma[idx]]
            
            # If we are in probability mode, then we must multiply with the probability of the
            # i-th row taking the value of the j-th candidate.
            if prob_mode:
                support *= probabilities[i][j]

            # Add the computed support to the winning class label.
            answer[winner] += support

    return answer

if __name__ == "__main__":
    
    # Instantiate the parser.
    description = "Answer the CP (certain prediction) queries for the K-Nearest Neighbor algorithm."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("query", type=str, choices=["q1", "q2"],
                        help="The CP query that should be answered. Options are: "
                             "q1, which is the checking query; and q2, which is the counting query.")

    parser.add_argument("dataset", type=str, 
                        help="Training dataset to use, given as a CSV file.")

    parser.add_argument("example", type=str, 
                        help="Test example to query for, given as a CSV file with a single line. "
                             "It should have the same column order as the training dataset CSV, "
                             "with the label column missing.")

    parser.add_argument("--k", type=int, default=1,
                        help="The K parameter of the KNN algorithm. (default=%d)" % DEFAULT_K)

    parser.add_argument("--label-col", type=int, default=0,
                        help="Integer representing the ordinal of the column to use as label. (default=0)")

    parser.add_argument("--kernel", type=str, choices=KERNELS.keys(), default="euclidean",
                        help="Integer representing the ordinal of the column to use as label. "
                             "(default=%s) (choices=%s)" % (DEFAULT_KERNEL, ", ".join(KERNELS.keys())))

    parser.add_argument("--no-header", action="store_true",
                        help="Assume that the training CSV file does not have a column header in the first line.")

    parser.add_argument("--prob-mode", action="store_true",
                        help="Used for the counting query (q2) to compute probabilities instead of counts.")

    args = parser.parse_args()

    # Parse the dataset CSV files.
    dataset = []
    dataset_probs = []
    labels = []
    dataset_col_domains = []
    dataset_col_probs = []
    with open(args.dataset, newline="") as f:
        reader = csv.reader(f)

        # Read and parse header if specified.
        if not args.no_header:
            header = next(reader)
            for col in header:
                d = re.search(r"\[(.+?)\]", col)
                if d:
                    domain_elements = [x.split("~") for x in d.group(1).split("|")]
                    dataset_col_domains.append([float(x[0].strip()) for x in domain_elements])
                    dataset_col_probs.append([float(x[1].strip()) if len(x) > 1 else 1.0 for x in domain_elements])

                    # Ensure probabilities sum to 1.
                    dataset_col_probs[-1] = [x / sum(dataset_col_probs[-1]) for x in dataset_col_probs[-1]]
                else:
                    dataset_col_domains.append([])
                    dataset_col_probs.append([])

        # Build the dataset out of cells which can contain incomplete values.
        for line in reader:
            row = []
            row_probs = []
            for i, cell in enumerate(line):
                if i == args.label_col:
                    labels.append(cell)
                else:
                    if cell == "":
                        assert(len(dataset_col_domains[i]) > 0)
                        row.append(dataset_col_domains[i])
                        row_probs.append(dataset_col_probs[i])
                    else:
                        d = re.search(r"\[(.+?)\]", cell)
                        if d:
                            domain_elements = [x.split("~") for x in d.group(1).split("|")]
                            row.append([float(x[0].strip()) for x in domain_elements])
                            row_probs.append([float(x[1].strip()) if len(x) > 1 else 1.0 for x in domain_elements])

                            # Ensure probabilities sum to 1.
                            row_probs[-1] = [x / sum(row_probs[-1]) for x in row_probs[-1]]
                        else:
                            row.append([float(cell)])
                            row_probs.append([float(1.0)])

            dataset.append(row)
            dataset_probs.append(row_probs)

    # Parse the example CSV file.
    with open(args.example, newline="") as f:
        reader = csv.reader(f)
        example = [float(x) for x in next(reader)]

    # Compute the similarity candidates.
    similarities = compute_similarities(dataset, example)

    # Answer the appropriate query.
    if args.query == "q1":
        answer = q1(similarities, labels, k=args.k)
    else:
        probabilities = compute_probabilities(dataset_probs) if args.prob_mode else None
        answer = q2(similarities, labels, k=args.k, probabilities=probabilities)

    # Print answer.
    print("class: answer")
    print("-------------")
    for k in sorted(answer.keys()):
        print(str(k) + ": " + str(answer[k]))
