from cards import Deck, Hand
from abc import ABC, abstractmethod

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

    def policy(self, opponent_hand: Hand):
        """
        Utilizes MonteCarlo methods to determine whether to hit or not.
        """

        # TODO: implement

        pass