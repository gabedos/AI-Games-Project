from cards import Deck
from agents import DealerAgent, QLearnAgent, MonteCarloAgent

class Game:
    def __init__(self, Player_agent, Dealer_agent, rounds=1):
        self.deck = Deck()
        self.Player_agent = Player_agent
        self.Dealer_agent = Dealer_agent
        self.rounds = rounds

    def play_round(self):

        self.deck.start_round()

        # Create players
        player = self.Player_agent(self.deck)
        dealer = self.Dealer_agent(self.deck)

        # Deal 3 cards
        # Note: not giving dealer 2 cards now because
        # it will provide extra information of unrevealed card

        c1 = self.deck.deal_card()
        c2 = self.deck.deal_card()
        c3 = self.deck.deal_card()

        player.hand.add_card(c1)
        player.hand.add_card(c2)

        dealer.hand.add_card(c3)

        # Player's turn
        while player.policy(dealer.hand):
            player.hit()

        # Player busted!
        if player.hand.value > 21:
            return 0

        c4 = self.deck.deal_card()
        dealer.hand.add_card(c4)

        # Dealer's turn
        while dealer.policy(player.hand):
            dealer.hit()

        # Determine winner
        return player.hand.compute_winner(dealer.hand)

    def start(self):

        player_wins = 0
        dealer_wins = 0
        ties = 0

        for _ in range(self.rounds):
            result = self.play_round()
            if result == 1:
                player_wins += 1
            elif result == 0:
                dealer_wins += 1
            else:
                ties += 1
        return player_wins/self.rounds, dealer_wins/self.rounds, ties/self.rounds

if __name__ == "__main__":

    game = Game(Player_agent=MonteCarloAgent, Dealer_agent=DealerAgent, rounds=100)

    outcomes = game.start()
    print(f"Player wins: {outcomes[0]*100}%")
    print(outcomes)
