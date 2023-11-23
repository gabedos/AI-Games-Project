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


class QLearnAgent(Agent):

    def policy(self, opponent_hand: Hand):
        """
        Utilizes Q-Learning to determine whether to hit or not.
        """

        # TODO: implement

        pass


class MonteCarloAgent(Agent):

    Explore_time = 1000

    def policy(self, opponent_hand: Hand):
        """
        Utilizes MonteCarlo methods to determine whether to hit or not.
        """

        start_time = time.time()
        start_state = BlackJackState(self.hand, opponent_hand, self.deck)

        root = MonteCarloNode(start_state, None)

        while time.time() - start_time < MonteCarloAgent.Explore_time:

            # Gets the leaf node in the UCB tree
            node = root.find_leaf_node()

            # Determines the random terminal value of the node
            reward = node.simulate()

            # Updates the rewards of the parents
            node.update_rewards(reward)

        node = root.get_best_average_child()

        return node.parent_action

class BlackJackState:

    def __init__(self, my_hand: Hand, dealer_hand: Hand, deck: Deck, last_hit: bool) -> None:
        self.my_hand = deepcopy(my_hand)
        self.dealer_hand = deepcopy(dealer_hand)
        self.deck = deepcopy(deck)

        # True if player stands
        self.stand = False

        # True if player previously hit
        self.last_move_hit = last_hit

    def _simulate_dealer(self):
        agent = DealerAgent(self.deck)
        while agent.policy(self.my_hand):
            agent.hit()

    def is_terminal(self):
        return self.my_hand.value > 21 or self.stand
    
    def get_actions(self):
        if self.last_move_hit:
            return self.deck.get_unique_cards()
        else:
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
            assert self.last_move_hit == False, "Cannot hit again without drawing cards"

            next_state.my_hand.add_card(next_state.deck.deal_card())

        elif action == "stand":
            next_state.stand = True
        else:

        return next_state


class MonteCarloNode:

    def __init__(self, state:BlackJackState, parent=None, parent_action=None):

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
	
        # assert len(self.missing_child_actions) > 0, "No more actions to expand"
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
        return current_node
    
    def simulate(self):
        """
        Returns the terminal value of the node by randomly simulating game.
        """

        payoff = self.state.find_terminal_value()
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
        Node: {self.state}
        Total Visits: {self.total_visits}
        Total Rewards: {self.total_rewards}
        Average Reward: {self.get_average_reward()}
        UCB Value: {ucb_value}
        Parent Action: {self.parent_action}
        Actor: {self.state.actor()}
        Missing Child Actions: {self.missing_child_actions}
        Children: {len(self.children)}
        """

        return string_form
