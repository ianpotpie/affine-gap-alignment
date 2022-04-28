import numpy as np


class ScoringScheme:
    def __init__(self, match_score=1.0, mismatch_penalty=-1.0, gap_penalty=-1.0, gap_start_penalty=0.0):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.gap_start_penalty = gap_start_penalty
        self.symbol_to_index = None
        self.scoring_matrix = None

    def get_symbols(self):
        """
        Creates a list of all symbols in the scoring matrix in the order in which they appear.
        Returns none if no scoring matrix has been loaded.

        :return: a list of symbols
        """
        if self.scoring_matrix is None:
            return None
        else:
            symbols = np.empty(len(self.symbol_to_index), dtype=str)
            for symbol, index in self.symbol_to_index.items():
                symbols[index] = symbol
            return list(symbols)

    def score(self, sequence1, sequence2):
        """
        Scores the alignment of two sequences based on their symbol-by-symbol scores going left-to-right.
        The method expects alignments to be the same length.
        It will quietly stop evaluating the score at the end of the shorter alignment.

        :param sequence1: the first sequence in the alignment to score
        :param sequence2: the second sequence in the alignment to score
        :return: the score of the alignment
        """
        score = 0.0
        for i, (symbol1, symbol2) in enumerate(zip(sequence1, sequence2)):

            if symbol1 == "-" and symbol2 == "-":
                raise ValueError("Encountered alignment between two gaps")

            elif symbol1 == "-":
                score += self.gap_penalty
                score += self.gap_start_penalty if (i > 0) and sequence1[i - 1] != "-" else 0.0

            elif symbol2 == "-":
                score += self.gap_penalty
                score += self.gap_start_penalty if (i > 0) and sequence2[i - 1] != "-" else 0.0

            elif self.scoring_matrix is not None:
                assert symbol1 in self.symbol_to_index, "Encountered symbol not in the current scoring matrix"
                assert symbol2 in self.symbol_to_index, "Encountered symbol not in the current scoring matrix"
                row = self.symbol_to_index[symbol1]
                col = self.symbol_to_index[symbol2]
                score += self.scoring_matrix[row][col]
            else:
                score += self.match_score if symbol1 == symbol2 else self.mismatch_penalty

        return score

    def __call__(self, sequence1, sequence2):
        """
        Calling the scoring scheme will score the two sequences passed in.
        You can find the behavior of the scoring in the "score" method.

        :param sequence1: the first sequence in the alignment to score
        :param sequence2: the second sequence in the alignment to score
        :return: the score of the alignment
        """
        return self.score(sequence1, sequence2)

    def load_matrix(self, filename):
        """
        Sets the scoring matrix of the scoring system based on the scoring matrix of a file.

        :param filename: the file containing the new matrix
        :return: None
        """
        with open(filename, mode='r') as f:
            headline = f.readline()
            while headline[0] == "#":  # iterates past all lines with comments
                headline = f.readline()
            self.symbol_to_index = {symbol.strip(): i for i, symbol in enumerate(headline.split())}

            # fill the scoring matrix
            n_symbols = len(self.symbol_to_index)
            self.scoring_matrix = np.zeros((n_symbols, n_symbols))
            for line in f:
                row = line.split()
                symbol = row.pop(0)
                i = self.symbol_to_index[symbol]
                for j, score in enumerate(row):
                    self.scoring_matrix[i, j] = float(score)

    def __str__(self):
        """
        Creates a string representation of the scoring scheme.
        If a scoring matrix is loaded, then it uses a standard PAM or BLOSUM style matrix.

        :return: a string of the scoring scheme
        """
        if self.scoring_matrix is None:
            s = f"Match Score: {self.match_score}\n" + \
                f"Mismatch Penalty: {self.mismatch_penalty}\n" + \
                f"Gap Start Penalty: {self.gap_start_penalty}\n" + \
                f"Gap Extension Penalty: {self.gap_penalty}"
        else:
            s = f"# Gap Start Penalty: {self.gap_start_penalty}\n" + \
                f"# Gap Extension Penalty: {self.gap_penalty}\n"

            symbols = self.get_symbols()
            s += "   " + "  ".join(symbols)
            for i in range(self.scoring_matrix.shape[0]):
                s += "\n" + symbols[i]
                for j in range(self.scoring_matrix.shape[1]):
                    score = self.scoring_matrix[i, j]
                    score = int(score) if score.is_integer() else score
                    s += f" {score}" if score < 0.0 else f"  {score}"

        return s
