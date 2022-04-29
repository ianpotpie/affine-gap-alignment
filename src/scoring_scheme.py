import numpy as np


class ScoringScheme:
    def __init__(self, match=1.0, mismatch=-1.0, gap=-1.0, gap_open=0.0, semi_global=False):
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.gap_open = gap_open
        self.semi_global = semi_global
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

    def score(self, seq1: str, seq2: str):
        """
        Scores the alignment of two sequences based on their symbol-by-symbol scores going left-to-right.
        The method expects alignments to be the same length.
        It will quietly stop evaluating the score at the end of the shorter alignment.

        :param seq1: the first sequence in the alignment to score
        :param seq2: the second sequence in the alignment to score
        :return: the score of the alignment
        """
        if self.semi_global:
            seq1, seq2 = seq1.lstrip("-"), seq2.lstrip("-")  # disregard gaps at the beginning
            seq1, seq2 = seq1[-min(len(seq1), len(seq2)):], seq2[-min(len(seq1), len(seq2)):]
            seq1, seq2 = seq1.rstrip("-"), seq2.rstrip("-")  # disregard gaps at the end
            seq1, seq2 = seq1[:min(len(seq1), len(seq2))], seq2[:min(len(seq1), len(seq2))]

        score = 0.0
        for i, (symbol1, symbol2) in enumerate(zip(seq1, seq2)):

            if symbol1 == "-" and symbol2 == "-":
                raise ValueError("Encountered alignment between two gaps")

            elif symbol1 == "-":
                score += self.gap
                score += self.gap_open if (i > 0) and seq1[i - 1] != "-" else 0.0

            elif symbol2 == "-":
                score += self.gap
                score += self.gap_open if (i > 0) and seq2[i - 1] != "-" else 0.0

            elif self.scoring_matrix is not None:
                if (symbol1 not in self.symbol_to_index) or (symbol2 not in self.symbol_to_index):
                    raise ValueError("Encountered a symbol not in the scoring matrix")
                row = self.symbol_to_index[symbol1]
                col = self.symbol_to_index[symbol2]
                score += self.scoring_matrix[row][col]
            else:
                score += self.match if symbol1 == symbol2 else self.mismatch

        return score

    def __call__(self, seq1, seq2):
        """
        Calling the scoring scheme will score the two sequences passed in.
        You can find the behavior of the scoring in the "score" method.

        :param seq1: the first sequence in the alignment to score
        :param seq2: the second sequence in the alignment to score
        :return: the score of the alignment
        """
        return self.score(seq1, seq2)

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
            s = f"Match Score: {self.match}\n" + \
                f"Mismatch Penalty: {self.mismatch}\n" + \
                f"Gap Start Penalty: {self.gap_start}\n" + \
                f"Gap Extension Penalty: {self.gap}"
        else:
            s = f"# Gap Start Penalty: {self.gap_start}\n" + \
                f"# Gap Extension Penalty: {self.gap}\n"

            symbols = self.get_symbols()
            s += "   " + "  ".join(symbols)
            for i in range(self.scoring_matrix.shape[0]):
                s += "\n" + symbols[i]
                for j in range(self.scoring_matrix.shape[1]):
                    score = self.scoring_matrix[i, j]
                    score = int(score) if score.is_integer() else score
                    s += f" {score}" if score < 0.0 else f"  {score}"

        return s
