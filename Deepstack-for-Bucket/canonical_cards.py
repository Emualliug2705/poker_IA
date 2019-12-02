import time
import collections


def canonical(cards, r, f):
    """
    Rules for a canonical hand:
    1. The cards are in sorted order

    2. The i-th suit must have at least many cards as all later suits.  If a
       suit isn't present, it counts as having 0 cards.

    3. If two suits have the same number of cards, the ranks in the first suit
       must be lower or equal lexicographically (e.g., [1, 3] <= [2, 4]).

    4. Must be a valid hand (no duplicate cards)
    """

    if sorted(cards) != cards:
        return False
    by_suits = collections.defaultdict(list)
    for suit in range(0, r * f, r):
        by_suits[suit] = [card % r for card in cards if suit <= card < suit + r]
        if len(set(by_suits[suit])) != len(by_suits[suit]):
            return False
    for suit in range(r, r * f, r):
        suit1 = by_suits[suit - r]
        suit2 = by_suits[suit]
        if not suit2: continue
        if len(suit1) < len(suit2):
            return False
        if len(suit1) == len(suit2) and suit1 > suit2:
            return False
    return True


def deal_cards(permutations, n, cards, r, f):
    if len(cards) == n:
        permutations.append(list(cards))
        return
    start = 0
    if cards:
        start = max(cards) + 1
    for card in range(start, r * f):
        cards.append(card)
        if canonical(cards, r, f):
            deal_cards(permutations, n, cards, r, f)
        del cards[-1]


def generate_permutations(n, r, f):

    permutations = []
    deal_cards(permutations, n, [], len(r), len(f))

    test = [['{0}{1}'.format(r[v % len(r)], f[v // len(r)])
        for v in c] for c in permutations ]
    test = tuple(map(tuple, test))
    return test


if __name__ == "__main__":
    from pypokertools.isomorph import *
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'][:6]
    flower = ['c', 'd', 's', 'h'][:2]

    state = ['Ac', 'Kc', 'Qc', 'Jc', 'Tc', '9c', 'Ad']
    ranks_ = list(set([v[0] for v in state]))
    flower_ = list(set([v[1] for v in state]))

    h = set(tuple(preflop_get_canonical(preflop)) for preflop in combinations([[v] for v in state], 2))
    h = generate_permutations(2, ranks_, flower_)

    from itertools import combinations
    h_ = generate_permutations(7, ranks, flower)

    all = tuple(combinations(state, 2))
    # new_state = [ranks.index(v[0]) * flower.index(v[1]) for v in state]

    h = generate_permutations(2, ranks, flower)

    """
    for i in [2, 3, 5, 7]:
        t0 = time.time()
        h = generate_permutations(i, ranks, flower)

        print('{}'.format(len(h)))
        print('before mapping : {}s'.format(time.time() - t0))
        t0 = time.time()
        print('before mapping : {}s'.format(time.time() - t0))
    """

"""
cards = create_CARDS(NUMERICAL_RANKS=range(2, 8), SUITS='dc')
t0 = time.time()
h = generate_permutations(2, cards)
print('{}'.format(time.time()-t0))
"""
