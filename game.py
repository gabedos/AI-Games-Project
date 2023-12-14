from cards import Deck
from agents import Agent, DealerAgent, QLearnAgent, MonteCarloAgent

class Game:
    def __init__(self, player_agent: Agent, player_args: dict, dealer_agent: Agent, rounds=1):
        self.deck = Deck()
        self.player_agent = player_agent
        self.player_args = player_args
        self.dealer_agent = dealer_agent
        self.rounds = rounds

    def play_round(self):

        self.deck.start_round()

        # Update players decks
        self.player_agent.deck = self.deck
        self.dealer_agent.deck = self.deck

        player = self.player_agent(self.deck, **self.player_args)
        dealer = self.dealer_agent(self.deck)

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

    ## Blackjack Background ##

    # The game being played is a simplified version of blackjack where the player and dealer are dealt 2 cards each initially.
    # The goal is to get as close to 21 as possible without going over. You are competing against the dealer!

    # The player is then given the option to hit or stand. If the player hits, they are dealt another card.
    # If the player stands, then their turn is over
    # The dealer then hits cards according to their pre-defined strategy

    ## MCTS Agent ##

    # First we will test the Monte Carlo agent against the dealer! One unique challenge was dealing with the fact that
    # the game of blackjack is stochastic. Originally, I tried using Chance Nodes but ran into many difficulties. Then,
    # I realized I could squash all the 13 different outcomes from hitting into one node and update the deck as we traverse
    # down the tree. This does has some additional overhead as a result of making deep copies of the deck at each node, but
    # that is what made the algorithm work throughout my various attempts.

    ## MCTS Agent Results ##

    # Under t = 0.005 second per move (averaging 5-10 simulations), the win rate was 39.51% when having 25,000 simulations!
    game = Game(player_agent=MonteCarloAgent, player_args={"explore_time": 0.005}, dealer_agent=DealerAgent, rounds=1000)
    outcomes = game.start()
    print(f"(M1: t=0.005) Player wins: {outcomes[0]*100}%", f"Dealer wins: {outcomes[1]*100}%", f"Ties: {outcomes[2]*100}%")

    # Under t = 0.1 second per move (20x more time), the win rate rose to 42.6% when having 10,000 simulations!
    game = Game(player_agent=MonteCarloAgent, player_args={"explore_time": 0.1},dealer_agent=DealerAgent, rounds=1000)
    outcomes = game.start()
    print(f"(M2: t=0.1) Player wins: {outcomes[0]*100}%", f"Dealer wins: {outcomes[1]*100}%", f"Ties: {outcomes[2]*100}%")

    # We notice that as we increase the time and number of simulations, the MCTS agent approaches the optimal strategy
    # With the maximum theoretical win rate of approximately 42.5%.

    # One interesting thing to note is that increasing the amount of time makes minimal improvements to the results.
    # That is, the agent given barely any time to think performs a few percentage points away from optimal game play!

    # NOTE: The agent given extra time may underperform the agent given less time due to the fact that less iterations
    # have been run so there is much higher variance which is especially influential when the different is only about 2%.
    # If you have extra time, try running the mcts agents for more rounds and see if the win rate converges towards 39.5% and 42.5%!