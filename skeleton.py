import random

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.value} of {self.suit}"

class Deck:
    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        self.cards = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()

    def __repr__(self):
        return f"Deck of {len(self.cards)} cards"

class Hand:
    def __init__(self):
        self.cards = []
        self.value = 0
        self.aces = 0

    def add_card(self, card):
        self.cards.append(card)
        self.value += self.card_value(card)
        self.adjust_for_ace()

    def card_value(self, card):
        if card.value.isnumeric():
            return int(card.value)
        elif card.value == 'Ace':
            self.aces += 1
            return 11
        else:
            return 10

    def adjust_for_ace(self):
        while self.value > 21 and self.aces:
            self.value -= 10
            self.aces -= 1

    def __repr__(self):
        return f"Hand: {' '.join(map(str, self.cards))} (Value: {self.value})"

class Agent:
    def __init__(self):
        self.hand = Hand()

    def policy(self):
        return self.hand.value < 17

    def hit(self, deck):
        self.hand.add_card(deck.deal_card())

class Game:
    def __init__(self, rounds=1):
        self.deck = Deck()
        self.player = Agent()
        self.dealer = Agent()
        self.rounds = rounds

    def play_round(self):
        # Initial deal
        for _ in range(2):
            self.player.hit(self.deck)
            self.dealer.hit(self.deck)

        # Player's turn
        while self.player.policy():
            self.player.hit(self.deck)

        # Dealer's turn
        while self.dealer.policy():
            self.dealer.hit(self.deck)

        # Determine winner
        player_bust = self.player.hand.value > 21
        dealer_bust = self.dealer.hand.value > 21
        player_win = self.player.hand.value > self.dealer.hand.value

        if player_bust:
            return "Dealer wins!"
        elif dealer_bust or player_win:
            return "Player wins!"
        else:
            return "Dealer wins!"

    def start(self):
        outcomes = []
        for _ in range(self.rounds):
            self.deck = Deck()  # Reshuffle the deck
            self.player = Agent()  # Reset player
            self.dealer = Agent()  # Reset dealer
            outcome = self.play_round()
            outcomes.append(outcome)
        return outcomes

# Example of running the game for 5 rounds
game = Game(rounds=500)
outcomes = game.start()
for outcome in outcomes:
    print(outcome)
