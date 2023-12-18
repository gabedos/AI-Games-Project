from cards import Deck
from agents import DealerAgent, QLearnAgent, MonteCarloAgent

class Game:
    def __init__(self, Player_agent, Dealer_agent, rounds=1, q_table=None):
        self.deck = Deck()
        self.Player_agent = Player_agent
        self.Dealer_agent = Dealer_agent
        self.rounds = rounds
        self.q_table = q_table

    def play_round(self):

        self.deck.start_round()
        # Create player
        # check if it is a QLearnAgent
        if self.Player_agent == QLearnAgent:
            player = self.Player_agent(self.deck, self.q_table)
        else:
            player = self.Player_agent(self.deck)
        dealer = self.Dealer_agent(self.deck)

        # if the player already has cards, reset them
        player.hand.reset()
        dealer.hand.reset()

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
            #Player busted!
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

        # if it is a QLearnAgent, train it first
        if self.Player_agent == QLearnAgent:
            # make a deck to use for training
            training_deck = Deck()
            training_player = self.Player_agent(training_deck)
            q_table = training_player.train(1000000) # train for 1 million rounds
            # training_player.print_q_table() # print the q_table
            self.q_table = q_table # set the q_table to the trained q_table


        for i in range(self.rounds):
            result = self.play_round()
            if result == 1:
                player_wins += 1
            elif result == 0:
                dealer_wins += 1
            else:
                ties += 1
        return player_wins/self.rounds, dealer_wins/self.rounds, ties/self.rounds

if __name__ == "__main__":

    game = Game(Player_agent=QLearnAgent, Dealer_agent=DealerAgent, rounds=5000000)

    outcomes = game.start()
    print(f"Player wins: {outcomes[0]*100}%")
    print(outcomes)

