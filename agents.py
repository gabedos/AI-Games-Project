import time
import math
import random
from cards import Deck, Hand
from abc import ABC, abstractmethod
from copy import deepcopy

class Agent(ABC):

    def __init__(self, deck: Deck):
        self.deck = deck
        self.hand = Hand()

    def hit(self):
        """
        Adds a card to the hand.
        """
        self.hand.add_card(self.deck.deal_card())

    @abstractmethod
    def policy(self, opponent_hand: Hand):
        """
        Implement a policy using the deck and opponent's hand.
        """
        pass


class DealerAgent(Agent):

    def policy(self, opponent_hand: Hand):
        """
        Base dealer policy: hits if value under 17.
        """
        return self.hand.value < 17


import random
from collections import defaultdict
import numpy as np

class QLearnAgent(Agent):
    def __init__(self, deck: Deck):
        super().__init__(deck)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001
        self.opponent_hand = None  # To store the opponent's hand

    def train(self, training_duration=10):
        start_time = time.time()
        while time.time() - start_time < training_duration:
            self.deck.deal_deck()
            self.hand = Hand()
            self.opponent_hand = Hand()  # Reset opponent hand for training
            self.hit()
            self.hit()  # Start with two cards

            while not self.hand.value > 21:
                state = self.get_state()
                if state not in self.q_table:
                    self.q_table[state] = {'hit': 0, 'stand': 0}

                action = self.choose_action(state)
                self.perform_action(action)

                new_state = self.get_state()
                reward = self.get_reward(new_state)
                self.update_q_table(state, action, reward, new_state)

                if action == 'stand' or self.hand.value > 21:
                    break

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(['hit', 'stand'])
        return max(self.q_table[state], key=self.q_table[state].get)

    def perform_action(self, action):
        if action == 'hit':
            self.hit()

    def get_reward(self, new_state):
        if new_state[0] > 21:  # Bust
            return -1
        return 0

    def update_q_table(self, state, action, reward, new_state):
        if new_state not in self.q_table:
            self.q_table[new_state] = {'hit': 0, 'stand': 0}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[new_state].values())
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate -= self.exploration_decay_rate

    def policy(self, opponent_hand: Hand):
        self.opponent_hand = opponent_hand  # Store the opponent's hand for use in get_state
        self.train(training_duration=1)  # Train for 10 seconds
        state = self.get_state()
        if state not in self.q_table:
            self.q_table[state] = {'hit': 0, 'stand': 0}
        return self.choose_action(state) == 'hit'

    def get_state(self):
        hand_value = self.hand.value
        has_ace = any(card.value == 'A' for card in self.hand.cards)
        is_hot = self.deck.heat > 0
        dealer_value = self.opponent_hand.value if self.opponent_hand.cards else 0
        return (hand_value, has_ace, is_hot, dealer_value)



class MonteCarloAgent(Agent):
    """
    MonteCarloAgent utilizes MonteCarlo methods to determine whether to hit or not.
    """

    # NOTE: 0.25 seconds may yield approximately 300 simulations depending on device
    Explore_time = 1

    def policy(self, opponent_hand: Hand):
        """
        Utilizes MonteCarlo methods to determine whether to hit or not.
        """

        start_time = time.time()
        start_state = BlackjackStateMCTS(self.hand, opponent_hand, self.deck)

        root = MonteCarloNode(start_state, None)

        while time.time() - start_time < MonteCarloAgent.Explore_time:

            # Gets the leaf node in the UCB tree
            node = root.find_leaf_node()

            # Determines the random terminal value of the node
            reward = node.simulate()

            # Updates the rewards of the parents
            node.update_rewards(reward)

        node = root.get_best_average_child()

        return True if node.parent_action == "hit" else False


class BlackjackStateMCTS:

    def __init__(self, my_hand: Hand, dealer_hand: Hand, deck: Deck) -> None:
        self.my_hand = deepcopy(my_hand)
        self.dealer_hand = deepcopy(dealer_hand)
        self.deck = deepcopy(deck)

        # True if player stands
        self.stand = False

    def _simulate_dealer(self):

        agent = DealerAgent(self.deck)
        while agent.policy(self.my_hand):
            agent.hit()

    def is_terminal(self):
        return self.my_hand.value > 21 or self.stand
    
    def get_actions(self):
        return ["hit", "stand"]

    def find_terminal_value(self):
        self._simulate_dealer()
        return self.my_hand.compute_winner(self.dealer_hand)
    
    def successor(self, action):
        """
        Returns the successor state of the current state given an action.
        """
        next_state = deepcopy(self)

        if action == "hit":
            next_state.my_hand.add_card(next_state.deck.deal_card())
        else:
            next_state.stand = True

        return next_state
    
    def refresh_hand(self, parent_deck: Deck, parent_hand: Hand):
        """
        Refreshes the hand by dealing a new card from the parent's deck.
        """
        self.deck = parent_deck
        self.my_hand = parent_hand
        self.my_hand.add_card(self.deck.deal_card())


class MonteCarloNode:

    def __init__(self, state:BlackjackStateMCTS, parent=None, parent_action=None):

        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        self.total_visits = 0
        self.total_rewards = 0

        self.children = []

        if self.state.is_terminal():
            self.missing_child_actions = []
        else:
            self.missing_child_actions = self.state.get_actions()

    def is_fully_expanded(self):
        """
        Determines if the node is fully expanded.
        """
        return len(self.missing_child_actions) == 0

    def get_average_reward(self):
        if self.total_visits == 0:
            return 0
        return self.total_rewards / self.total_visits

    def get_best_average_child(self):
        """
        Returns the child node with the best average reward.
        """
        node = max(self.children, key = lambda x: x.get_average_reward())
        return node

    def get_ucb_value(self, parent_total_visits):
        """
        Returns the UCB value of the node using parents visits and actor
        """
        value = self.get_average_reward() + math.sqrt(2 * math.log(parent_total_visits) / self.total_visits)
        return value

    def get_best_ucb_child(self):
        """
        Returns the child node with the best UCB value.
        """
        node = max(self.children, key = lambda x: x.get_ucb_value(self.total_visits))
        return node
    
    def expand(self):
        """
        Expands the node by adding a new child node from an unexplored action.
        """
        action = self.missing_child_actions.pop()
        next_state = self.state.successor(action)
        child_node = MonteCarloNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def find_leaf_node(self):
        """
        Returns the leaf node of the tree using UCB values.
        """
        current_node = self
        while not current_node.state.is_terminal():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:

                current_node = current_node.get_best_ucb_child()
                
                # Refresh node by dealing a new card from parent's deck
                parent_deck = deepcopy(current_node.parent.state.deck)
                parent_hand = deepcopy(current_node.parent.state.my_hand)
                current_node.state.refresh_hand(parent_deck, parent_hand)

        return current_node
    
    def simulate(self):
        """
        Returns the terminal value of the node by randomly simulating game.
        """
        state = self.state
        while not state.is_terminal():
            state = state.successor(random.choice(state.get_actions()))

        payoff = state.find_terminal_value()
        return payoff
    
    def update_rewards(self, reward):
        """
        Updates the total reward and total visits of the node and all its parents.
        """
        parent_node = self
        while parent_node is not None:
            parent_node.total_rewards += reward
            parent_node.total_visits += 1
            parent_node = parent_node.parent

    def __str__(self) -> str:

        ucb_value = "N/A"
        if self.parent:
            ucb_value = self.get_ucb_value(self.parent.total_visits)

        string_form = f"""
        Total Visits: {self.total_visits}
        Total Rewards: {self.total_rewards}
        Average Reward: {self.get_average_reward()}
        UCB Value: {ucb_value}
        Parent Action: {self.parent_action}
        """

        return string_form
