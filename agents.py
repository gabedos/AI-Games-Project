import time
import math
import random
from cards import Deck, Hand
from abc import ABC, abstractmethod
from copy import deepcopy

class Agent(ABC):

    def __init__(self, deck: Deck = None):
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
        number = random.random()
        if number > 0.5:
            return True
        else:
            return False


class MonteCarloAgent(Agent):
    """
    MonteCarloAgent utilizes MonteCarlo methods to determine whether to hit or not.
    """

    def __init__(self, deck: Deck = None, **kwargs):
        super().__init__(deck)
        self.explore_time = kwargs.get("explore_time", 0.005)

    def policy(self, opponent_hand: Hand):
        """
        Utilizes MonteCarlo methods to determine whether to hit or not.
        """

        start_state = BlackjackStateMCTS(self.hand, opponent_hand, self.deck)

        # Edge case: if player has >= 21 then game over
        if start_state.is_terminal():
            return False

        root = MonteCarloNode(start_state, None)

        start_time = time.time()
        while time.time() - start_time < self.explore_time:

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

    def is_terminal(self):
        return self.my_hand.value >= 21 or self.stand

    def get_actions(self):
        return ["hit", "stand"]

    def find_terminal_value(self):

        # Create dealer agent with same hand!
        dealer = DealerAgent(deepcopy(self.deck))
        dealer.hand = deepcopy(self.dealer_hand)

        # Simulate dealer's turn
        while dealer.policy(self.my_hand):
            dealer.hit()

        # Determine winner
        winner = self.my_hand.compute_winner(dealer.hand)
        return winner
    
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
    
    def refresh_cards(self, parent_deck: Deck, parent_hand: Hand, parent_action: str):
        """
        Refreshes the cards in the game state.
        """
        self.deck = parent_deck
        self.my_hand = parent_hand
        if parent_action == "hit":
            self.my_hand.add_card(self.deck.deal_card())

    def __str__(self) -> str:
        string_form = f""" ---
        Hand: {self.my_hand}
        Stand: {self.stand}
        """
        return string_form


class MonteCarloNode:

    def __init__(self, state:BlackjackStateMCTS, parent=None, parent_action=None):

        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        self.total_visits = 0
        self.total_rewards = 0

        self.children = []
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

                # Refresh node by building a fresh hand from parent node
                current_node = current_node.get_best_ucb_child()

                current_node.state.refresh_cards(
                    deepcopy(current_node.parent.state.deck),
                    deepcopy(current_node.parent.state.my_hand), 
                    current_node.parent_action)

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
        State: {self.state}
        Children: {self.children}
        IsFullyExpanded: {self.is_fully_expanded()}
        """

        return string_form
