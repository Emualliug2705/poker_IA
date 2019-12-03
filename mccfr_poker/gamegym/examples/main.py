import os

import numpy as np
import pytest

from gamegym.algorithms import (BestResponse, OutcomeMCCFR, RegretStrategy, approx_exploitability,
                                exploitability)
from gamegym.algorithms.stats import sample_payoff
from gamegym.games import (DicePoker, Goofspiel, MatchingPennies, MatrixZeroSumGame,
                           RockPaperScissors)
from gamegym.strategy import UniformStrategy

g = DicePoker()
mc = OutcomeMCCFR(g, seed=52)
mc.compute(10000, burn=0.5)
