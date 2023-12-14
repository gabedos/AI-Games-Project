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

    print("Welcome to Blackjack! executing simulations in game.py\n")

    ## Blackjack Background ##

    # The game being played is a simplified version of blackjack where the player and dealer are dealt 2 cards each initially.
    # The goal is to get as close to 21 as possible without going over. You are competing against the dealer!

    # The player is then given the option to hit or stand. If the player hits, they are dealt another card.
    # If the player stands, then their turn is over
    # The dealer then hits cards according to their pre-defined strategy

    ## MCTS Agent ## By Gabriel Dos Santos

    # First we will test the Monte Carlo agent against the dealer! One unique challenge was dealing with the fact that
    # the game of blackjack is stochastic. Originally, I tried using Chance Nodes to represent "hit" states in which a 
    # card would be drawn. Then, the Chance Node would randomly sample a node based on its probability distribution. However,
    # I ran into difficulties trying to get the tree to work with 2 different types of nodes within it.

    # Then, I realized I could replace the Chance Node & its children by squashing all 13 different outcomes from hitting into one node.
    # That is, instead of picking a node based on the probability distribution, I could just draw a card from the deck
    # (which inherently models the probabilty distribution) and update the live deck as we traverse down the tree.
    # This results in each mcts tree node's game state slightly changing as we traverse as a result of different cards being randomly drawn,
    # but they still consistently represent hit or stand nodes as originally created. One drawback of this implementation is that we need 
    # to create deepcopies of the deck each iteration so that we have the same starting deck each time!

    ## MCTS Agent Results ##

    print("MCTS Agent vs Dealer\n", "-"*20)
    ROUNDS = 1000

    # Under t = 0.005 second per move (averaging 5-10 simulations), the win rate was 39.51% when having 25,000 simulations!
    game = Game(player_agent=MonteCarloAgent, player_args={"explore_time": 0.005}, dealer_agent=DealerAgent, rounds=ROUNDS)
    outcomes = game.start()
    print("(M1: t=0.005)\t",
          f"Player wins: {round(outcomes[0]*100, 3)}%",
          f"Dealer wins: {round(outcomes[1]*100, 3)}%",
          f"Ties: {round(outcomes[2]*100, 3)}%")

    # Under t = 0.1 second per move (20x more time), the win rate rose to 42.6% when having 10,000 simulations!
    game = Game(player_agent=MonteCarloAgent, player_args={"explore_time": 0.1},dealer_agent=DealerAgent, rounds=ROUNDS)
    outcomes = game.start()
    print("(M1: t=0.1)\t",
          f"Player wins: {round(outcomes[0]*100, 3)}%",
          f"Dealer wins: {round(outcomes[1]*100, 3)}%",
          f"Ties: {round(outcomes[2]*100, 3)}%")

    # We notice that as we increase the time and number of simulations, the MCTS agent approaches the optimal strategy
    # with the maximum theoretical win rate of approximately 42.5%.

    # One interesting thing to note is that increasing the amount of time makes minimal improvements to the results.
    # That is, the agent given barely any time to think performs a few percentage points away from optimal game play!

    # NOTE: The test script only runs 1000 iterations so there is a much higher variance. Since the difference in win rate is only about 3%,
    # there is a chance that the extra time doesn't make a difference in the test script. If you have time, try increasing the number of rounds!

    ## Q-Learning Agent ## By EJ Wilford

    print("\nQ-Learning Agent vs Dealer\n", "-"*20)
