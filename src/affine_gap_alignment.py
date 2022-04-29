import sys
import argparse
import numpy as np
from scoring_scheme import ScoringScheme


def global_alignment(seq1: str, seq2: str, scoring_scheme):
    """
    The Gotoh algorithm can be used to perform global alignment according to the provided scoring scheme.

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
            M[i, j] = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + scoring_scheme(seq1[i - 1], seq2[j - 1])

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


def semiglobal_alignment(seq1: str, seq2: str, scoring_scheme):
    """
    The Gotoh algorithm can be used to perform semi-global alignment according to the provided scoring scheme.

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
    X[0, :] = 0.0
    X[:, 0] = -np.inf

    Y = np.empty((max_i + 1, max_j + 1), dtype=float)
    Y[:, 0] = 0.0
    Y[0, :] = -np.inf

    # populate the dynamic programming matrix
    for i in range(1, max_i + 1):
        for j in range(1, max_j + 1):
            M[i, j] = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + scoring_scheme(seq1[i - 1], seq2[j - 1])

            if i < max_i:
                X[i, j] = max(M[i, j - 1] + gap_open + gap_extend,
                              X[i, j - 1] + gap_extend,
                              Y[i, j - 1] + gap_open + gap_extend)
            else:
                X[i, j] = max(M[i, j - 1], X[i, j - 1], Y[i, j - 1])

            if j < max_j:
                Y[i, j] = max(M[i - 1, j] + gap_open + gap_extend,
                              X[i - 1, j] + gap_open + gap_extend,
                              Y[i - 1, j] + gap_extend)
            else:
                Y[i, j] = max(M[i - 1, j], X[i - 1, j], Y[i - 1, j])

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
                    if (i < max_i and X[i, j] == M[i, j - 1] + gap_open + gap_extend) or X[i, j] == M[i, j - 1]:
                        next_backtraces.append((alignment1, alignment2, "M", i, j - 1))
                    if (i < max_i and X[i, j] == X[i, j - 1] + gap_extend) or X[i, j] == X[i, j - 1]:
                        next_backtraces.append((alignment1, alignment2, "X", i, j - 1))
                    if (i < max_i and X[i, j] == Y[i, j - 1] + gap_open + gap_extend) or X[i, j] == Y[i, j - 1]:
                        next_backtraces.append((alignment1, alignment2, "Y", i, j - 1))

                if matrix == "Y":
                    alignment1, alignment2 = seq1[i - 1] + alignment1, "-" + alignment2
                    if (j < max_j and Y[i, j] == M[i - 1, j] + gap_open + gap_extend) or Y[i, j] == M[i - 1, j]:
                        next_backtraces.append((alignment1, alignment2, "M", i - 1, j))
                    if (j < max_j and Y[i, j] == X[i - 1, j] + gap_open + gap_extend) or Y[i, j] == X[i - 1, j]:
                        next_backtraces.append((alignment1, alignment2, "X", i - 1, j))
                    if (j < max_j and Y[i, j] == Y[i - 1, j] + gap_extend) or Y[i, j] == Y[i - 1, j]:
                        next_backtraces.append((alignment1, alignment2, "Y", i - 1, j))

        backtraces = next_backtraces

    return max_score, alignments


def local_alignment(seq1: str, seq2: str, scoring_scheme):
    """
    The Gotoh algorithm can be used to perform local alignment according to the provided scoring scheme.

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
    X[0, :] = 0.0
    X[:, 0] = -np.inf

    Y = np.empty((max_i + 1, max_j + 1), dtype=float)
    Y[:, 0] = 0.0
    Y[0, :] = -np.inf

    # populate the dynamic programming matrix
    best_score, best_locations = 0.0, []
    for i in range(1, max_i + 1):
        for j in range(1, max_j + 1):
            substitution = scoring_scheme(seq1[i - 1], seq2[j - 1])
            M[i, j] = max(M[i - 1, j - 1] + substitution,
                          X[i - 1, j - 1] + substitution,
                          Y[i - 1, j - 1] + substitution,
                          0.0)

            X[i, j] = max(M[i, j - 1] + gap_open + gap_extend,
                          X[i, j - 1] + gap_extend,
                          Y[i, j - 1] + gap_open + gap_extend,
                          0.0)

            Y[i, j] = max(M[i - 1, j] + gap_open + gap_extend,
                          X[i - 1, j] + gap_open + gap_extend,
                          Y[i - 1, j] + gap_extend,
                          0.0)

            if M[i, j] > best_score:
                best_score, best_locations = M[i, j], []
            if M[i, j] == best_score:
                best_locations.append(("M", i, j))

    # backtrace to find the alignments that created the max alignment score
    alignments = []
    backtraces = [("", "", matrix, i, j) for matrix, i, j in best_locations]
    while len(backtraces) > 0:
        next_backtraces = []
        for alignment1, alignment2, matrix, i, j in backtraces:
            if (matrix == "M" and M[i, j] == 0.0) \
                    or (matrix == "X" and X[i, j] == 0.0) \
                    or (matrix == "Y" and Y[i, j] == 0.0):
                alignments.append((alignment1, alignment2))
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

    return best_score, alignments


def main():
    parser = argparse.ArgumentParser(description="Align two sequences.")
    parser.add_argument("sequence1", type=str)
    parser.add_argument("sequence2", type=str)
    parser.add_argument("--gap-open", type=float, default=-2)
    parser.add_argument("--gap", type=float, default=-1)
    parser.add_argument("--match", type=float, default=1)
    parser.add_argument("--mismatch", type=float, default=-1)
    parser.add_argument("--matrix", type=str)
    parser.add_argument("--type", choices=["global", "local", "semiglobal"], type=str, default="global")
    args = parser.parse_args(sys.argv[1:])

    # the first sequence to align
    sequence1 = args.sequence1

    # the second sequence to align
    sequence2 = args.sequence2

    # create a scoring scheme
    scoring_scheme = ScoringScheme()
    scoring_scheme.gap_open = args.gap_open
    scoring_scheme.gap = args.gap
    scoring_scheme.match = args.match
    scoring_scheme.mismatch = args.mismatch
    if args.matrix is not None:
        scoring_scheme.load_matrix(args.matrix)

    wall_width = 33
    if args.type == "local":
        max_score, alignments = local_alignment(sequence1, sequence2, scoring_scheme)
        print("Gotoh Affine-Gap Local Alignment")
        wall_width = 32
    elif args.type == "semiglobal":
        max_score, alignments = semiglobal_alignment(sequence1, sequence2, scoring_scheme)
        print("Gotoh Affine-Gap Semi-Global Alignment")
        wall_width = 38
    else:  # args.type == "global":
        max_score, alignments = global_alignment(sequence1, sequence2, scoring_scheme)
        print("Gotoh Affine-Gap Global Alignment")
        wall_width = 33

    print(wall_width * "-")
    print(f"Sequence 1: {sequence1}")
    print(f"Sequence 2: {sequence2}")
    print(f"Optimal Alignment Score: {max_score}")
    print()
    print("Optimal Alignments:")
    print(wall_width * "-")
    for i, alignment in enumerate(alignments):
        print(alignment[0])
        print(alignment[1])
        print(wall_width * "-")


if __name__ == "__main__":
    main()
