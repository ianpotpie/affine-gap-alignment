import sys
import argparse
import numpy as np
from scoring_scheme import ScoringScheme


def global_alignment(seq1: str, seq2: str, scoring_scheme):
    """
    The Gotoh algorithm can be used to perform global alignment according to the provided scoring scheme.
    The algorithm does implement affine gap alignment.

    :param seq1: the first sequence to align
    :param seq2: the second sequence to align
    :param scoring_scheme: the scoring system used in the alignment
    :return: a max alignment score and a list of alignments that produce that score
    """
    # initialize constants
    gap_open = scoring_scheme.gap_open
    gap_extend = scoring_scheme.gap
    max_i = len(seq1)
    max_j = len(seq2)

    # initialize the scoring matrices
    M = np.empty((max_i + 1, max_j + 1), dtype=float)
    M[0, :] = -np.inf
    M[:, 0] = -np.inf
    M[0, 0] = 0.0

    X = np.empty((max_i + 1, max_j + 1), dtype=float)
    X[0, :] = gap_open + gap_extend * np.arange(max_j + 1)
    X[:, 0] = -np.inf

    Y = np.empty((max_i + 1, max_j + 1), dtype=float)
    Y[:, 0] = gap_open + gap_extend * np.arange(max_i + 1)
    Y[0, :] = -np.inf

    # populate the dynamic programming matrix
    for i in range(1, max_i + 1):
        for j in range(1, max_j + 1):
            M[i, j] = max(M[i - 1, j - 1],
                          X[i - 1, j - 1],
                          Y[i - 1, j - 1]) + scoring_scheme(seq1[i - 1], seq2[j - 1])

            X[i, j] = max(M[i, j - 1] + gap_open + gap_extend,
                          X[i, j - 1] + gap_extend,
                          Y[i, j - 1] + gap_open + gap_extend)

            Y[i, j] = max(M[i - 1, j] + gap_open + gap_extend,
                          X[i - 1, j] + gap_open + gap_extend,
                          Y[i - 1, j] + gap_extend)

    max_score = max(M[max_i, max_j], X[max_i, max_j], Y[max_i, max_j])

    # backtrace to find the alignments that created the max alignment score
    alignments = []
    backtraces = []  # each backtrace has the form (alignment1, alignment2, matrix, i, j)

    if M[max_i, max_j] == max_score:
        backtraces.append(("", "", "M", max_i, max_j))

    if X[max_i, max_j] == max_score:
        backtraces.append(("", "", "X", max_i, max_j))

    if Y[max_i, max_j] == max_score:
        backtraces.append(("", "", "Y", max_i, max_j))

    while len(backtraces) > 0:
        next_backtraces = []
        for alignment1, alignment2, matrix, i, j in backtraces:
            if i == 0:
                alignments.append(((j * "-") + alignment1, seq2[:j] + alignment2))
            elif j == 0:
                alignments.append((seq1[:i] + alignment1, (i * "-") + alignment2))
            else:
                if matrix == "M":
                    alignment1, alignment2 = seq1[i - 1] + alignment1, seq2[j - 1] + alignment2
                    if M[i, j] == M[i - 1, j - 1] + scoring_scheme(seq1[i - 1], seq2[j - 1]):
                        next_backtraces.append((alignment1, alignment2, "M", i - 1, j - 1))
                    if M[i, j] == X[i - 1, j - 1] + scoring_scheme(seq1[i - 1], seq2[j - 1]):
                        next_backtraces.append((alignment1, alignment2, "X", i - 1, j - 1))
                    if M[i, j] == Y[i - 1, j - 1] + scoring_scheme(seq1[i - 1], seq2[j - 1]):
                        next_backtraces.append((alignment1, alignment2, "Y", i - 1, j - 1))

                if matrix == "X":
                    alignment1, alignment2 = "-" + alignment1, seq2[j - 1] + alignment2
                    if X[i, j] == M[i, j - 1] + gap_open + gap_extend:
                        next_backtraces.append((alignment1, alignment2, "M", i, j - 1))
                    if X[i, j] == X[i, j - 1] + gap_extend:
                        next_backtraces.append((alignment1, alignment2, "X", i, j - 1))
                    if X[i, j] == Y[i, j - 1] + gap_open + gap_extend:
                        next_backtraces.append((alignment1, alignment2, "Y", i, j - 1))

                if matrix == "Y":
                    alignment1, alignment2 = seq1[i - 1] + alignment1, "-" + alignment2
                    if Y[i, j] == M[i - 1, j] + gap_open + gap_extend:
                        next_backtraces.append((alignment1, alignment2, "M", i - 1, j))
                    if Y[i, j] == X[i - 1, j] + gap_open + gap_extend:
                        next_backtraces.append((alignment1, alignment2, "X", i - 1, j))
                    if Y[i, j] == Y[i - 1, j] + gap_extend:
                        next_backtraces.append((alignment1, alignment2, "Y", i - 1, j))

        backtraces = next_backtraces

    return max_score, alignments


def main():
    parser = argparse.ArgumentParser(description="Align two sequences.")
    parser.add_argument("sequence1", type=str)
    parser.add_argument("sequence2", type=str)
    parser.add_argument("--gap-open", type=float)
    parser.add_argument("--gap", type=float)
    parser.add_argument("--match", type=float)
    parser.add_argument("--mismatch", type=float)
    parser.add_argument("--matrix", type=str)
    args = parser.parse_args(sys.argv[1:])

    # the first sequence to align
    sequence1 = args.sequence1

    # the second sequence to align
    sequence2 = args.sequence2

    # create a scoring scheme
    scoring_scheme = ScoringScheme()
    if args.gap_open is not None:
        scoring_scheme.gap_open = -abs(args.gap_open)
    if args.gap is not None:
        scoring_scheme.gap = -abs(args.gap)
    if args.match is not None:
        scoring_scheme.match = args.match
    if args.mismatch is not None:
        scoring_scheme.mismatch = -abs(args.mismatch)
    if args.matrix is not None:
        scoring_scheme.load_matrix(args.matrix)

    max_score, alignments = global_alignment(sequence1, sequence2, scoring_scheme)

    print("Gotoh Affine-Gap Global Alignment")
    print(33 * "-")
    print(f"Sequence 1: {sequence1}")
    print(f"Sequence 2: {sequence2}")
    print(f"Optimal Alignment Score: {max_score}")
    print()
    print("Optimal Alignments:")
    print(33 * "-")
    for i, alignment in enumerate(alignments):
        print(alignment[0])
        print(alignment[1])
        print(33 * "-")


if __name__ == "__main__":
    main()
