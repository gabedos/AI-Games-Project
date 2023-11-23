import random

class Card:
    def __init__(self, suit: str, value: str):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.value} of {self.suit}"

class Deck:

    Deck_num = 6
    Redeal_percentage = 0.5
    Suits = ['H', 'D', 'C', 'S']
    Values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    Low_cards = set(['2', '3', '4', '5', '6'])
    High_cards = set(['10', 'J', 'Q', 'K', 'A'])

    def __init__(self):
        self.deal_deck()

    def start_round(self):
        """
        Checks if the deck needs to be reshuffled and reshuffles if necessary.
            Note: should only execute at the start of each game's round
        """
        if len(self.cards) > len(Deck.Suits) * len(Deck.Values) * Deck.Deck_num * Deck.Redeal_percentage:
            self.deal_deck()

    def deal_card(self) -> Card :
        """
        Returns a card from the deck and removes it from the deck.
        """

        removed_card = self.cards.pop()
        if removed_card.value in Deck.Low_cards:
            self._heat -= 1
        elif removed_card.value in Deck.High_cards:
            self._heat += 1

        return removed_card

    def deal_deck(self):
        self.cards = [Card(suit, value) for suit in Deck.Suits for value in Deck.Values] * Deck.Deck_num
        random.shuffle(self.cards)
        self._heat = 0

    @property
    def heat(self):
        """
        Returns the average heat of the deck.
        """
        number_of_decks = len(self.cards) / (len(Deck.Suits) * len(Deck.Values))
        average_heat = self._heat / number_of_decks
        return average_heat

    def __repr__(self):
        return f"Deck of {len(self.cards)} cards"

class Hand:

    def __init__(self):
        self.cards = []
        self.value = 0
        self.aces = 0

    def add_card(self, card: Card):
        """
        Adds a card to the hand and updates the value of the hand.
        """
        self.cards.append(card)
        self.update_value(card)

    def update_value(self, card: Card):
        """
        Updates the value of the hand.
        """

        # Determine card value
        card_value = 0
        if card.value.isnumeric():
            card_value = int(card.value)
        elif card.value == 'A':
            card_value = 11
            self.aces += 1
        else:
            card_value = 10
        self.value += card_value

        # Adjust for aces
        while self.value > 21 and self.aces:
            self.value -= 10
            self.aces -= 1

    def compute_winner(self, dealer_hand):
        """
        Computes the winner of the game.
        """
        player_bust = self.value > 21
        dealer_bust = dealer_hand.value > 21
        player_win = self.value > dealer_hand.value

        if player_bust:
            return 0
        elif dealer_bust or player_win:
            return 1
        else:
            # Ties are losses for the player
            return 0

    def __repr__(self):
        return f"Hand: {' '.join(map(str, self.cards))} (Value: {self.value})"