# AI-Games-Project
#### By Gabriel Dos Santos (gds6) and EJ Wilford (ejw58)
Implementation of Blackjack with various intelligent algorithms for determining optimal play.

## Monte Carlo Agent

### Results

Under t = 0.005 second per move (averaging 5-10 simulations), the win rate was 39.51% under 25,000 simulations!
Under t = 0.1 second per move (20x more time), the win rate rose to xx.x% under xx,xxx simulations! **INSERT**

### Monte Carlo Tree Search (MCTS) Agent

#### Approach
- The MCTS Agent overcomes the stochastic nature of Blackjack, originally planned to use Chance Nodes, by incorporating a method that draws a card from the deck to model the probability distribution. This approach required deep copies of the deck for each iteration to maintain consistency.

#### Results
- **Performance Under Time Constraint (t=0.005 seconds per move):**
  - With an average of 5-10 simulations, the win rate was 39.51% over 25,000 simulations.
- **Performance with Increased Time (t=0.1 seconds per move):**
  - Allowing 20x more time, the win rate increased to 42.6% over 10,000 simulations.
- **Insights:**
  - Increasing the number of simulations and time per move, the MCTS agent approaches the optimal strategy with a theoretical maximum win rate of approximately 42.5%.
  - The agent performs competitively even with minimal thinking time, suggesting efficiency in strategy formulation.

#### Considerations
- Variance in results due to limited iterations (1000 rounds).
- Recommendation: Increase the number of rounds to validate the impact of additional time on performance.

## Q-Learning Agent

#### Approach
- Utilizes an epsilon-greedy policy for exploration and learning.
- State space includes player’s hand value, dealer’s hand value, presence of a usable ace, and deck heat for pseudo-card counting.
- Q-table initialized with zeros and updated via formula with class.

#### Results
- **Performance with Limited Training (1000 rounds):**
  - The agent achieved a win rate of 37.5%.
- **Performance with Extensive Training (100,000 rounds):**
  - The win rate improved to 40.7%, a modest increase despite a significant increase in training rounds.
- **Insights:**
  - Shows better performance with more training rounds due to increased experience and learning.
  - Hyperparameter tuning might help in reaching closer to the optimal win rate (~43%).

#### Considerations
- Similar to the MCTS agent, high variance in results due to limited test iterations.
- Suggestion: Experiment with the number of rounds for a more comprehensive performance comparison between the two agents.


