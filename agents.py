import time
import math
import random
from cards import Deck, Hand
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict
import numpy as np


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
    def __init__(self, deck, q_table=None, alpha=0.3, gamma=0.9, epsilon=.2, alpha_decay=0.999):
        """
        Initialize the Q-learning agent.
        """
        super().__init__(deck)
        self.q_table = q_table if q_table is not None else self.create_q_table() # Initialize Q-table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha_decay = alpha_decay

    def create_q_table(self): #DONE
        """
        create a Q-table with all possible states and actions
        """
        q_table = defaultdict(lambda: defaultdict(float))
        for dealer_card in range(1, 12):
            for player_value in range(1, 22):
                for player_has_ace in [True, False]:
                    for deck_heat in ['hot', 'nuetral','cold']:
                        state = (dealer_card, player_value, player_has_ace, deck_heat)
                        q_table[state]['hit'] = 0
                        q_table[state]['stay'] = 0
        return q_table

    def print_q_table(self): #DONE
        """
        Print the current Q-table
        """
        #do the above programatically
        for player_has_ace in [True, False]:
            for deck_heat in ['hot', 'nuetral', 'cold']:
                print("Q-Table for: " + deck_heat + " and " + ("Ace" if player_has_ace else "No Ace"))
                print("Dealer Card: ", end=' ')
                print()
                for i in range(1, 11):
                    print(i, end=' ')
                print()
                for player_value in range(11, 22):
                    print(player_value, end=' ')
                    for dealer_card in range(1, 11):
                        state = (dealer_card, player_value, player_has_ace, deck_heat)
                        print('H' if self.q_table[state]['hit'] > self.q_table[state]['stay'] else 'S', end=' ')
                    print()
                print()

    def policy(self, opponent_hand): #DONE
        """
        Decide the action (hit or stay) based on the Q-table for the given state
        """
        # print the length of the hand
        # print(len(self.hand.cards))
        state = self._get_state(opponent_hand)
        # if the state is new, we randomly choose an action
        if state not in self.q_table:
            return random.choice(['hit', 'stay'])
        return  True if self.q_table[state]['hit'] > self.q_table[state]['stay'] else False

    def train(self, rounds): #DONE
        """
        Train the agent for a given number of rounds
        """
        for _ in range(rounds):
            self.deck.start_round()
            self._play_round()
            self.alpha = self.alpha * self.alpha_decay

        self.print_q_table()
        return self.q_table

    def _play_round(self):
        """
        Play a single round for training
        """
        # Setting up the hand for the round
        self.hand = Hand()
        opponent_hand = Hand()
        self.hand.add_card(self.deck.deal_card())
        self.hand.add_card(self.deck.deal_card())
        opponent_hand.add_card(self.deck.deal_card())
        
        # Get the initial state
        initial_pos = self._get_state(opponent_hand)
        is_done = False
        while not is_done: # while we haven't busted or stayed
            action = self._choose_action(initial_pos) # choose an action
            new_state, is_done = self._take_action(action, opponent_hand) # take the action
            reward = 0
            if is_done: # if we're done, we get the reward from the new state
                future_reward = self._get_intermediate_reward(initial_pos, action, new_state, opponent_hand) 
            else: # if we're not done, we get the max reward from the new state
                future_reward = max(self.q_table[new_state].values())
            # update the q_table
            result = self.q_table[initial_pos][action]
            self.q_table[initial_pos][action] = (1 - self.alpha) * result + self.alpha * (reward + self.gamma * future_reward) # update the q_table using the equation from class
            initial_pos = new_state # set the new state to the initial state


    def _choose_action(self, state):
        """
        Choose action based on the current state using an epsilon-greedy strategy
        """
        if random.random() < self.epsilon or state not in self.q_table:  # Epsilon-greedy exploration
            return random.choice(['hit', 'stay'])
        else:  # Exploitation
            return 'hit' if self.q_table[state]['hit'] > self.q_table[state]['stay'] else 'stay'

    def _take_action(self, action, opponent_hand):
        """
        Take the chosen action and find out the reward and new state
        """
        if action == 'hit':
            self.hit()
        new_state = self._get_state(opponent_hand) # get us the new state from hitting
        is_done = (self.hand.value > 21) or (action == 'stay') # if we bust or stay, we're done
        return new_state, is_done
    
    def _get_intermediate_reward(self, state, action, new_state, opponent_hand):
        """
        Get the intermediate reward for taking the given action and transitioning to the new state
        """
        # if we bust, we get a reward of -1
        if new_state[1] > 21:
            return -1
        # if we stay, we get a reward of 0
        elif action == 'stay':
            # we need to play out the dealer's turn to get the reward
            while opponent_hand.value < 17:
                opponent_hand.add_card(self.deck.deal_card())
            outcome = self._determine_outcome(opponent_hand)
            reward = self._get_reward(outcome)
            return reward
        # if we hit and get 21, we get a reward of 1
        elif new_state[1] == 21:
            return 1
        # if we hit and don't bust or get 21, we get a reward of 0
        else:
            return 0

    def _get_reward(self, outcome):
        """
        calc the reward based on the game outcome- only use in the END of the game
        """
        if outcome == 'win':
            # print("We win")
            return 1
        elif outcome == 'lose':
            # print("Womp womp womp we lost")
            return -1
        else:  # tie
            # print("womp womp we tie")
            return 0

    def _determine_outcome(self, opponent_hand):
        """
        Determine the outcome of the game based on the final hand values of the player and opponent. could merge with get_reward/replace with compute_winner
        """
        if self.hand.value > 21: # if we bust, we lose
            return 'lose'
        elif opponent_hand.value > 21 or self.hand.value > opponent_hand.value: # if the dealer busts or we have a higher value, we win
            return 'win'
        elif self.hand.value < opponent_hand.value: # if the dealer has a higher value, we lose
            return 'lose'
        else:
            return 'tie'

    def _get_state(self, opponent_hand): # THIS WORKS
        """
        set the current state based on the game context
        """
        dealer_value = opponent_hand.value
        player_value = self.hand.value
        player_has_ace = 'A' in [card.value for card in self.hand.cards]
        # print(len(self.hand.cards))
        if self.deck.heat < -3:
            deck_heat = 'cold'
        elif self.deck.heat > 3:
            deck_heat = 'hot'
        else:
            deck_heat = 'nuetral'
        return (dealer_value, player_value, player_has_ace, deck_heat)




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
