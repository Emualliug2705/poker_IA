# -*- coding:utf-8 -*-

''' The number of hand suit'''
suit_count = 4

''' The number of rank suit'''
rank_count = 13

''' Directory to write results'''
file_path = 'data/'

''' Use heuristic'''
dict_heuristic = {'flop': False, 'turn': False, 'river':False}

''' Nb Clusters for value'''
dict_clusters = {'flop': 800, 'turn': 4800, 'river': 5000}

''' The number of cards we want to play'''
card_count = suit_count * rank_count

''' The number of player cards cluster in river'''
river_cluster_count = 5

''' The number of player cards cluster in turn'''
turn_cluster_count = 5

''' The number of player cards cluster in flop'''
flop_cluster_count = 80

''' The number of public cards in four rounds'''
board_card_count = [0, 3, 4, 5]

''' The number of player hold cards '''
hold_card_count = 2
