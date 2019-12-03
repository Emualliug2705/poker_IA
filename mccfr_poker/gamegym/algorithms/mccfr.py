import pandas as pd
import logging
import time

import numpy as np

from gym_env.env import Stage, CommunityData, PlayerData, PlayerCycle, StageData
from ..situation import Situation, Action
from ..strategy import Strategy
from ..utils import debug_assert, get_rng, np_uniform

from gym_env.env import PlayerCycle
log = logging.getLogger(__name__)


winner_in_episodes = []

STOP_LCFR = 0.2


class RegretStrategy(Strategy):
    """
    Average strategy and regret storage for one player.

    Dictionary based, any unknown observations return uniform distributions.
    """
    EPS = 1e-30

    def __init__(self):
        # observation: (regrets, strategy)
        self.table = {}
        # usage statistics
        self.queries = 0
        self.misses = 0
        # training statistics
        self.updates = 0
        self.iterations = 0

    def get_entry(self, observation: tuple, actions: int) -> tuple:
        assert isinstance(observation, tuple)
        entry = self.table.get(observation, None)
        if entry is None:
            entry = (np.zeros(actions), np.zeros(actions))
        else:
            assert len(entry[0]) == actions
            assert len(entry[1]) == actions
        return entry

    def update_entry(self, observation: tuple, actions: int, dr=None, ds=None) -> tuple:
        assert isinstance(observation, tuple)
        entry = self.table.get(observation, None)
        if entry is None:
            entry = (np.zeros(actions), np.zeros(actions))

        nr = (entry[0] + dr) if dr is not None else entry[0]
        ns = (entry[1] + ds) if ds is not None else entry[1]
        self.table[observation] = (nr, ns)
        self.updates += 1

    def regret_matching(self, regret):
        "Return stratefy distribution based on the regret vector"
        regplus = np.clip(regret, 0, None)
        s = np.sum(regplus)
        if s > self.EPS:
            return regplus / s
        else:
            return np_uniform(len(regret))

    def _strategy(self, observation, n_actions: int, situation: Situation = None) -> tuple:
        self.queries += 1
        entry = self.table.get(observation, None)
        if entry is not None and np.sum(entry[1]) > self.EPS:
            dist = entry[1] / np.sum(entry[1])
        else:
            dist = np_uniform(n_actions)
            self.misses += 1
        assert n_actions == len(dist)
        return dist


class MCCFRBase:
    """
    Common base for Outcome and External sampling MC CFR.
    """

    def __init__(self, game, strategies=None, update=None, seed=None, rng=None):
        self.game = game
        self.rng = get_rng(rng, seed)
        self.strategies = strategies
        game.num_of_players = len(game.players)

        if self.strategies is None:
            self.strategies = tuple(RegretStrategy() for i in range(game.num_of_players))
        assert len(self.strategies) == game.num_of_players
        self.update = update
        if self.update is None:
            self.update = tuple(range(game.num_of_players))
        for i in self.update:
            assert isinstance(self.strategies[i], RegretStrategy)
        # stats
        self.iterations = 0
        self.nodes_traversed = 0

    def compute(self, iterations, epsilon=0.6, weight=1.0, progress=True, burn=0.0):
        """
        Run MC CFR for given iterations.

        Optionally uses a progress bar (default on).

        Updates to the cummulative strategy and regret are weighted by `weight`, this
        allows you to discount early iterations.
        If `burn > 0.0`, perform smooth burn-in by multiplying `weight` by a coefficient
        going from 0.03 to 1.0 in the first `burn`-fraction of iterations (off by default,
        a sensible choice is e.g. 0.3).
        """
        log = logging.getLogger('gamegym_original.MCCFR')
        log.debug("Computing {} for {} (iterations={}, weight={:.4g}, epsilon={:.4g})".format(
            self.__class__.__name__, repr(self.game), iterations, weight, epsilon))
        its = range(iterations)
        if progress:
            import tqdm
            its = tqdm.tqdm(its, desc="MCCFR")
        if burn > 0.0:
            assert burn <= 1.0

        for i in its:
            self.iterations += 1
            if i < burn * iterations:
                # This was tuned on Goofspiel(4); range 0.01-0.1 seems to be reasonable.
                w = 0.03**(1.0 - float(i) / iterations / burn)
            else:
                w = 1.0
            if progress:
                r = "nodes: {}".format(self.nodes_traversed)
                if burn > 0.0:
                    r += ", burn-in: {:.2f}".format(w)
                its.set_postfix_str(r)
            times_passed = list()
            for player in self.update:  # For each players
                t0 = time.time()
                self.sampling(player, epsilon=epsilon, weight=w * weight)
                t1 = time.time()-t0
                times_passed.append(t1)
                # print(f"Time passed {t1}s - Average time {sum(times_passed)/len(times_passed)}s")

            # When only one player is updated, we need to traverse as a dummy player
            # to update the cumulative strategies.
            # Using player -1 as the updated one is a bit hacky but works.
            if len(self.update) == 1:
                self.sampling(-1, epsilon=epsilon, weight=w * weight)

    def sampling(self, player, epsilon=0.6, weight=1.0):
        "Run one sampling run for the given player."
        raise NotImplementedError


class OutcomeMCCFR(MCCFRBase):
    def _outcome_sampling(self, situation, player_updated, p_reach_updated, p_reach_others,
                          p_sample, epsilon, weight):
        """
        Based on Alg 3 from PhD_Thesis_MarcLanctot.pdf and cfros.cpp from his bluff11.zip.
        Returns `(utility, p_tail, p_sample_leaf)`.
        """
        self.nodes_traversed += 1

        if situation.is_terminal():
            situation._end_hand()
            situation._calculate_reward()
            situation.distributed = False
            return situation.payoff[player_updated], 1.0, p_sample

        if situation.is_chance():
            # ai = self.rng.choice(len(situation.actions), p=situation.chance)
            # sit2 = self.game.play(situation, situation.actions[ai])

            # IF Chance NODE
            if not situation.distributed:
                situation.stage = Stage.PREFLOP
                situation._distribute_cards()  # Distribue les cartes à chaque joueurs
                situation._initiate_round()  # Initialise le round
                situation.distributed = True
            else:
                situation._end_round()
                situation._initiate_round()

            situation.previous_stage = situation.stage
            # No need to factor in the chances in Outcome sampling
            return self._outcome_sampling(situation, player_updated, p_reach_updated, p_reach_others,
                                          p_sample, epsilon, weight)

        # Extract misc, read entry from storage
        player = situation.player_cycle.idx  # Which is the next player to play
        situation._get_environment()
        strat = self.strategies[player]  # strat adopted by the player
        obs = situation.obs[player]  # Tuple cast from list to change ; Obs du jeu (carte en main)
        actions = situation.legal_moves  # Actions possible pour joueur actuel
        # actions = situation.current_player.agent_obj.action(situation.legal_moves, situation.observation, situation.info) # Tuple ('check', 'raise') for example

        # Treat static players as chance nodes
        if player not in self.update:
            dist = strat.strategy(situation)
            ai = self.rng.choice(len(situation.actions), p=dist)
            sit2 = self.game.play(situation, situation.actions[ai])
            # No need to factor in the chances in Outcome sampling
            return self._outcome_sampling(sit2, player_updated, p_reach_updated, p_reach_others,
                                          p_sample, epsilon, weight)

        # Create dists, sample the action
        entry = strat.get_entry(obs, len(actions))
        dist = strat.regret_matching(entry[0])

        # exploration in self-actions
        if player == player_updated:
            dist_sample = dist * (1.0 - epsilon) + 1.0 * epsilon / len(actions)
        else:
            dist_sample = dist
        debug_assert(lambda: np.abs(np.sum(dist) - 1.0) < 1e-3)
        debug_assert(lambda: np.abs(np.sum(dist_sample) - 1.0) < 1e-3)

        # Actions (bigblind, smallblind, call, fold, raise)
        action_idx = self.rng.choice(len(actions), p=dist_sample)
        action = actions[action_idx]

        # Future situation
        situation._execute_step(action)  # Execute l'action
        # if situation.first_action_for_hand[player] or situation.done:
        #     situation.first_action_for_hand[player] = False
        #     situation._calculate_reward(action)

        if player == player_updated:
            # Update regret / current strategy
            payoff, p_tail, p_sample_leaf = self._outcome_sampling(
                situation, player_updated, p_reach_updated * dist[action_idx], p_reach_others,
                p_sample * dist_sample[action_idx], epsilon, weight)
            dr = np.zeros_like(entry[0])
            U = payoff * p_reach_others / p_sample_leaf
            for ai in range(len(actions)):
                if ai == action_idx:
                    dr[ai] = U * (p_tail - p_tail * dist[action_idx])
                else:
                    dr[ai] = -U * p_tail * dist[action_idx]
            strat.update_entry(obs, len(actions), dr=dr * weight)
        else:
            # Update cumulative strategy
            payoff, p_tail, p_sample_leaf = self._outcome_sampling(
                situation, player_updated, p_reach_updated, p_reach_others * dist[action_idx],
                p_sample * dist_sample[action_idx], epsilon, weight)
            ds = (p_reach_others / p_sample) * dist
            strat.update_entry(obs, len(actions), ds=ds * weight)

        return payoff, p_tail * dist[action_idx], p_sample_leaf

    def sampling(self, updated_player, epsilon=0.6, weight=1.0):
        """
        Run one outcome sampling for the given player.
        """
        if updated_player >= 0:
            self.strategies[updated_player].iterations += 1
        
        """Reset after game over."""
        self.game.observation = None
        self.game.reward = None
        self.game.info = None
        self.game.done = False
        self.game.funds_history = pd.DataFrame()
        self.game.first_action_for_hand = [True] * len(self.game.players)

        for player in self.game.players:
            player.stack = self.game.initial_stacks

        self.game.dealer_pos = 0
        self.game.player_cycle = PlayerCycle(self.game.players,
                                             dealer_idx=-1,
                                             max_steps_after_raiser=3 * len(self.game.players))

        # New hand
        self.game.time_0 = time.time()

        """Deal new cards to players and reset table states."""
        self.game._save_funds_history()
        if self.game._check_game_over():
            return

        self.game.table_cards = []
        self.game._create_card_deck()  # Créer le deck
        self.game.stage = Stage.CHANCE  # Définie le premier stage

        # preflop round1,2, flop>: round 1,2, turn etc...
        self.game.stage_data = [StageData(len(self.game.players)) for _ in range(8)]

        # pots
        self.game.community_pot = 0
        self.game.current_round_pot = 0
        self.game.player_pots = [0] * len(self.game.players)
        self.game.player_max_win = [0] * len(self.game.players)
        self.game.last_player_pot = 0
        self.game.played_in_round = 0
        self.game.first_action_for_hand = [True] * len(self.game.players)

        for player in self.game.players:
            player.cards = []

        # Define the dealer
        self.game._next_dealer()
        self.game.obs = ((),)*(self.game.num_of_players + 1)
        self._outcome_sampling(self.game, updated_player, 1.0, 1.0, 1.0, epsilon=epsilon, weight=weight)
