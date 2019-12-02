# -*- coding:utf-8 -*-
from __future__ import division
import gc
import copy
from tqdm import tqdm
from ..pypokertools.isomorph import *
from ..pypokertools.pokertools import *
from Cluster_data import *
from card_to_string_conversion import CARD_TO_STRING
from judging import judging
import settings
from Cluster_result import *

'''
Traverse all possible hands and generate a hand representation
@param street the round name
@param file_path the relative path for storing data
@class Gen_data
'''
class Gen_data(Cmatrix):
    def __init__(self,
                street=None,
                file_path='data/',
                ):

        super().__init__(street, file_path)
        self.savename = self.get_savename()
        assert street in ['river', 'turn', 'flop'], 'The parameter street is error'

        if street == 'river':
            pass
        elif street == 'turn':
            self.centroids_5 = self.load_data('river_cluster.csv')
        else:
            self.centroids_5 = self.load_data('river_cluster.csv')
            self.centroids_4 = self.load_data('turn_cluster.csv')
            self.matrix = self.get_Euclidean_Matrix(self.centroids_5)

        card_to_string = CARD_TO_STRING()
        self.card = card_to_string.rank_table[:settings.rank_count]
        self.flower = card_to_string.suit_table[:settings.suit_count]
        self.all_cards = [i+j for i in self.card for j in self.flower]
        self.all_hands_dict_to_can = None

    '''Cluster center point save file name'''
    def get_savename(self):
        savenames = {'river': self.file_path + 'river_data.csv',
                    'turn':  self.file_path + 'turn_data.csv',
                    'flop':  self.file_path + 'flop_data.csv'}
        return savenames.get(self.street)

    '''
    Traverse all possible opponents' hands and calculate our winning percentage in the current state
    @param free_cards possible hand cards pool
    @param hand player's current hand cards
    @param public Current 5 public cards
    return int win rate
    '''
    def win_rate_compute(self, all_opponent, hand, public):
        win_rate = [0 for _ in range(3)]

        # Watch this
        n = len(all_opponent)
        for opponent in all_opponent:
            win_rate[judging(hand, opponent, public)] += 1./n

        win_rate = win_rate[0] + win_rate[2] / 2.0
        return win_rate

    '''Generate data in river round'''
    def data_generator_5(self, heuristic=True):
        skipped = 0
        play = 0
        f = open(self.savename, 'wt')

        all_state = combinations(self.all_cards, 7)
        # all_state = list(map(list, combinations(self.all_cards, 7)))
        # all_state = generate_permutations(7, self.card, self.flower)
        # all_state = get_all_canonicals(self.all_cards, 7)
        for state in tqdm(all_state, desc='{}'.format(self.street)):
            all_hand = list(combinations(state, 2))
            if heuristic:
                all_hand_can_p = set([self.all_hands_dict_to_can[' '.join(h)] for h in all_hand])
            else:
                all_hand_can_p = all_hand

            # all_hand = list(list(preflop_get_canonical(preflop)) for preflop in combinations([v for v in state], 2))
            for hand, can_hand in zip(all_hand, all_hand_can_p):
                play += 1
                if heuristic:
                    can_hand = can_hand.split(' ')
                # hand_card = hand[0] + ' ' + hand[1]
                public_card = [card for card in state
                                if card not in hand]

                # Eject state with intersection btw hand and board
                if len(set(can_hand).intersection(public_card)) > 0:
                    skipped += 1
                    continue

                # Combinations all free cards hands for opponents
                free_cards = [card for card in self.all_cards
                                if card not in state]
                if heuristic:
                    all_hands_can_opponent_ = set([self.all_hands_dict_to_can[' '.join(h)] for h in list(map(list, combinations(free_cards, 2)))])

                    all_hands_can_opponent = [h.split(' ') for h in all_hands_can_opponent_\
                                              if len(set(h.split(' ')).intersection(public_card)) == 0\
                                              and len(set(h.split(' ')).intersection(can_hand)) == 0]
                else:
                    all_hands_can_opponent = list(map(list, combinations(free_cards, 2)))

                if len(all_hands_can_opponent) == 0:
                    skipped += 1
                    continue

                win_rate = self.win_rate_compute(all_hands_can_opponent, can_hand, public_card)
                if win_rate is None:
                    skipped += 1
                    continue
                to_str = str(win_rate)
                f.write(to_str + '\n')
        f.close()
        print('skipped {} - play {} - %({})'.format(skipped,play,  (play-skipped)/play))

    '''Generate data in turn round'''
    def data_generator_4(self, heuristic=True):
        skipped = 0
        play = 0
        f = open(self.savename, 'wt')
        all_state = combinations(self.all_cards, 6)
        # all_state = generate_permutations(6, self.card, self.flower)
        for state in tqdm(all_state, desc='{}'.format(self.street)):
            all_hand = list(map(list, combinations(state, 2)))
            if heuristic:
                all_hand_can_p = set([self.all_hands_dict_to_can[' '.join(h)] for h in all_hand])
            else:
                all_hand_can_p = all_hand

            for hand, can_hand in zip(all_hand, all_hand_can_p):
                if heuristic:
                    can_hand = can_hand.split(' ')

                public_card_turn = [card for card in state
                                    if card not in hand]
                free_cards = [card for card in self.all_cards
                                if card not in state]

                n_turn_count = len(free_cards)
                cha = [0 for _ in range(len(self.centroids_5))]
                for public_card_river in free_cards:
                    play += 1
                    public_card_board = public_card_turn + [public_card_river]
                    all_opponent_card = copy.deepcopy(free_cards)
                    all_opponent_card.remove(public_card_river)

                    # Eject state with intersection btw hand and public_card_board
                    if len(set(can_hand).intersection(public_card_board)) > 0:
                        skipped += 1
                        continue

                    # Combinations all free cards hands for opponents
                    if heuristic:
                        all_opponent_card = [card for card in self.all_cards
                                      if card not in state]
                        all_hands_can_opponent_ = set(
                            [self.all_hands_dict_to_can[' '.join(h)] for h in list(map(list, combinations(all_opponent_card, 2)))])
                        all_hands_can_opponent = [h.split(' ') for h in all_hands_can_opponent_ \
                                                  if len(set(h.split(' ')).intersection(public_card_board)) == 0 \
                                                  and len(set(h.split(' ')).intersection(can_hand)) == 0]
                    else:
                        all_opponent_card = [card for card in all_opponent_card
                                      if card not in state]
                        all_hands_can_opponent = list(map(list, combinations(all_opponent_card, 2)))

                    if len(all_hands_can_opponent) == 0:
                        skipped += 1
                        continue

                    win_rate = self.win_rate_compute(all_hands_can_opponent, can_hand, public_card_board)
                    if win_rate is None:
                        skipped += 1
                        continue
                    cha[np.argmin(list(map(lambda x: abs(win_rate - x[0]), self.centroids_5)))] += 1. / n_turn_count

                cha = list(map(str, cha))
                to_str = ','.join(cha)
                f.write(to_str + '\n')
        f.close()
        print('skipped {} - play {} - %({})'.format(skipped,play,  (play-skipped)/play))

    '''Generate data in flop round'''
    def data_generator_3(self, heuristic=True):
        skipped = 0
        play = 0
        f = open(self.savename, 'wt')
        all_state = combinations(self.all_cards, 5)
        for state in tqdm(all_state, desc='{}'.format(self.street)):

            all_hand = list(map(list, combinations(state, 2)))
            if heuristic:
                all_hand_can_p = set([self.all_hands_dict_to_can[' '.join(h)] for h in all_hand])
            else:
                all_hand_can_p = all_hand

            history = list()
            for hand, can_hand in zip(all_hand, all_hand_can_p):
                if heuristic:
                    can_hand = can_hand.split(' ')

                public_card_flop = [card for card in state
                                    if card not in hand]

                # Set history of board
                if public_card_flop in history:
                    continue
                history.append(public_card_flop)
                free_cards = [card for card in self.all_cards
                                if card not in hand and card not in public_card_flop]  # Check with full deck
                n_flod_count = len(free_cards)
                cha_2 = [0.] * len(self.centroids_4)
                for public_card_turn in free_cards:
                    public_cards_4 = public_card_flop + [public_card_turn]
                    free_cards_2 = copy.deepcopy(free_cards)
                    free_cards_2.remove(public_card_turn)

                    n_turn_count = len(free_cards_2)
                    cha = [0. for _ in range(len(self.centroids_5))]
                    for public_card_river in free_cards_2:
                        play += 1
                        public_card_board = public_cards_4 + [public_card_river]
                        all_opponent_card = copy.deepcopy(free_cards_2)
                        all_opponent_card.remove(public_card_river)

                        # Eject state with intersection btw hand and board
                        if len(set(can_hand).intersection(public_card_board)) > 0:
                            skipped += 1
                            continue

                        # Combinations all free cards hands for opponents
                        if heuristic:
                            all_hands_can_opponent_ = set(
                                [self.all_hands_dict_to_can[' '.join(h)] for h in
                                 list(map(list, combinations(all_opponent_card, 2)))])

                            all_hands_can_opponent = [h.split(' ') for h in all_hands_can_opponent_ \
                                                      if len(set(h.split(' ')).intersection(public_card_board)) == 0 \
                                                      and len(set(h.split(' ')).intersection(can_hand)) == 0]
                        else:
                            all_hands_can_opponent = list(map(list, combinations(all_opponent_card, 2)))

                        if len(all_hands_can_opponent) == 0:
                            skipped += 1
                            continue

                        win_rate = self.win_rate_compute(all_hands_can_opponent, can_hand, public_card_board)
                        if win_rate is None:
                            continue
                        index = np.argmin(list(map(lambda x: abs(win_rate - x[0]), self.centroids_5)))

                        cha[index] += 1. / n_turn_count
                    distance_list = list(map(lambda x: emd(np.array(cha), np.array(x), self.matrix), self.centroids_4))

                    min_distance_index = np.argmin(distance_list)
                    cha_2[min_distance_index] += 1 / n_flod_count
                to_str = ','.join(list(map(str, cha_2)))
                f.write(to_str + '\n')
        f.close()
        print('skipped {} - play {} - %({})'.format(skipped, play,  (play-skipped)/play))

    def generate_all_can_hands(self):
        print('Generate all the hand for the preflop')
        all_hand = list(map(list, combinations(self.all_cards, 2)))
        temp_hands = [list(preflop_get_canonical(h)) for h in all_hand]
        self.all_hands_dict_to_can = {' '.join(keys): ' '.join(value) for keys, value in zip(all_hand, temp_hands)}
        del all_hand, temp_hands
        gc.collect()

        # print('Generate all the hand for the flop')
        all_flops = list(map(list, combinations(self.all_cards, 3)))
        temp_flops = [list(flop_get_canonical(h)) for h in all_flops]
        self.all_flops_dict = {' '.join(keys): ' '.join(value) for keys, value in zip(all_flops, temp_flops)}
        del all_flops, temp_flops
        gc.collect()

    '''Main function for generating data'''
    def generate_data(self, heuristic=True):
        if heuristic:
            # Generate dict. from hand to can. hand
            self.generate_all_can_hands()

        # print('\nGenerate Data for the cluster: {0}'.format(self.street))
        if self.street == 'river':
            self.data_generator_5(heuristic=heuristic)
        if self.street == 'turn':
            self.data_generator_4(heuristic=heuristic)
        if self.street == 'flop':
            self.data_generator_3(heuristic=heuristic)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", type=str, default='river')
    parser.add_argument("--file_path", type=str, default='data/')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    for s in ['river', 'turn', 'flop']:
        data = Gen_data(street=s, file_path=settings.file_path)
        data.generate_data(heuristic=settings.dict_heuristic)
        cluster_data_main(street=s, cluster_k=settings.dict_clusters[s])

"""
Generate all the hand for the preflop
river: 916it [00:00, 909.80it/s]1.0936872959136963s
river: 11440it [00:14, 784.06it/s]
skipped 76950 - play 126016 - %(0.3893632554596242)
Generate all the hand for the preflop
turn: 8008it [01:34, 84.80it/s] 
skipped 444838 - play 727840 - %(0.3888244669158057)
Generate all the hand for the preflop
flop: 4368it [25:06,  2.90it/s]
skipped 2056904 - play 3342240 - %(0.38457322035521085)
"""

"""
R6F2
river: 792it [00:13, 58.31it/s]
skipped 0 - play 16632 - %(1.0)
turn: 924it [01:17, 11.92it/s]
skipped 0 - play 83160 - %(1.0)
flop: 792it [04:57,  2.67it/s]
skipped 0 - play 332640 - %(1.0)

"""