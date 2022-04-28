# When should you berserk in a 3+0 lichess arena tournament game?
Your goal in a lichess arena is to accumulate the most points possible. A win gives 2 points, a draw 1, and a loss 0. However, if you are on a win streak of 2 or more games, then you earn double points (so that a win gives 4 points and a draw 2) until you fail to win a game. Additionally, at the start of any game, you can choose to berserk: you cut the time on your clock in half, but if you win you get an extra point. (See the [lichess arena FAQ](https://lichess.org/tournament/help?system=arena) for details.) Of course, having less time should generally make winning more difficult. So the question arises: what policy should one follow to maximize their points in an arena tournament given this unique payoff structure? For example, you might think to play more conservatively and rarely berserk, focusing on prolonging win streaks to get double points as often as possible. Or you might attempt to play as aggressively as possible and berserk often, reasoning that the extra points you get from berserk game wins will outweigh the double points you'll lose out on when you do end up losing win streaks. I've attempted to give a rudimentary answer to this question here.

I considered only 1+0 and 3+0 tournaments for this analysis, which are the standard time controls for Bullet and SuperBlitz arenas, as well as for the monthly lichess Titled Arena and Blitz Titled Arena. This document considers the 3+0 case. From this point, the text in this document is the same as in the one considering the 1+0 case; only the plots differ.

## The data
Refer to this repo's README  and [pgn2csv](https://github.com/jmviz/pgn2csv) for some information on how and what data I collected. In short, I looked at all relevant lichess arena tournament games from 2017-2021. First, we attempt to answer the question of this document using only the empirical data. 

### Empirical expected value of berserking vs. not berserking
We first look at the relative frequencies of wins, draws, and losses (WDL) when a player berserks vs. when they do not berserk, for each matchup of player rating and opponent rating (looking only at games where the opponent does not berserk). Using the empirical relative frequencies as WDL probability estimates, we can multiply them by the respective points the player would earn in each result, and thereby get empirical expected values for both berserking and not berserking. For example, for a certain opponent rating matchup, a player's empirical WDL when not berserking might be `[0.5, 0.05, 0.45]`. Assuming they are not on a streak, their point payoffs for win, draw, and loss would be `[2, 1, 0]`. Then the expected value would be `0.5 * 2 + 0.05 * 1 + 0.45 * 0 = 1.05`. The player's empirical WDL when berserking might be `[0.35, 0.05, 0.6]`, while the point payoffs for win, draw, and loss would be `[3, 1, 0]`, so that the expected value would be `0.35 * 3 + 0.05 * 1 + 0.6 * 0 = 1.1`. So based on these empirical WDL probability estimates, it would be better to berserk in this case, since the expected value for berserking is greater than the expected value for not berserking. On the other hand, if the player is on a 2+ win streak, then the expected values for not berserking and berserking would be `0.5 * 4 + 0.05 * 2 + 0.45 * 0 = 2.1` and `0.35 * 5 + 0.05 * 2 + 0.6 * 0 = 1.85`, respectively. In this case, it would be better not to berserk, since the expected value for not berserking is greater than the expected value for berserking. 

This is what I have done in this plot. The color of the square in each rating matchup reflects the difference of the expected value of berserking and not berserking; the redder the color, the better it is for the player to berserk their opponent, and the bluer the color, the better it is not to berserk. The size of each square corresponds to the number of games in each rating matchup in which one player berserked. I clamped the colormap to 1 and -1 so that outlier values with a very small amount of games did not cause the colormap to lose resolution. 

As one might expect, the lower your opponent's rating compared to yours, the better it is to berserk. Also, it seems that if your opponent's rating is near yours, berserking is less promising, and this is compounded if you are on a streak. If your opponent's rating is much greater than yours, the picture is less clear: the differences in expected value are generally much slighter.

![3_empirical_berserk_expected_value_difference](/plots/3_empirical_berserk_expected_value_difference.png)

### Policy for berserking vs. not berserking based on empirical expected values
Taking the calculations from the previous plot, we can generate a simple policy: if the empirical expected value for berserking is greater than the empirical expected value for not berserking, then berserk; otherwise, don't berserk. This plot and others like it below should be interpreted with caution as the degree to which it is better or worse to berserk is lost here. Also, the noise due to the paucity of games in which a player berserked for some rating matchups is more evident here. 

![3_empirical_berserk_expected_value_better](/plots/3_empirical_berserk_expected_value_better.png)

## Modeling the data
I fit a series of models on different aspects of the data, with an eye to generating a more informed policy for berserking than the one described above. This more detailed policy is shown in the final section of this notebook.

### Modeling WDL probabilities 
I fit a logistic regression model to the data to predict the WDL probabilities for every combination of `(white rating, black rating, white berserked, black berserked)`. The fitting was done in [wdl_model_search.py](/wdl_model_search.py). Using the same calculations as described above, one can take these model WDL probabilities and using them to generate model expected values for (not) berserking in different scenarios. Here we see that the model expected values generally agree well with empirical expected values. The only deficiency seems to be some trouble predicting the empirical expected value curves at the lowest player rating levels.  

![3_expected_value_curves.gif](/plots/3_expected_value_curves.gif)

For completion's sake, we can also redo the two plots above, this time using the model WDL probabilities. 

### WDL model expected value of berserking vs. not berserking
We see the model generates a similar pattern to the empirical one. I have likewise clamped the colormap here to 1 and -1. 

![3_berserk_expected_value_difference.png](/plots/3_berserk_expected_value_difference.png)

### Policy for berserking vs. not berserking based on WDL model expected values
The model policy has smoothed out the noise of the empirical one, but otherwise generated a similar pattern. The same proviso applies here: the degree to which berserking is better than not is lost in this plot.

![3_berserk_expected_value_better.png](/plots/3_berserk_expected_value_better.png)

### Modeling the probability that your opponent berserks
It will be important below to have some model of the probability that your opponent berserks, given various factors. Unfortunately, I neglected to record some of the relevant factors when I was processing the PGNs, and other relevant factors are impossible to discern from the PGNs (e.g., if both sides berserked, which side berserked first?). Therefore, I used a crude approach in which I fit another logistic regression model to the data which predicts your opponent's probability of berserking based on your rating and what color each player has (and assuming you yourself have not decided yet whether to berserk). This fitting was done in [berserk_model_search.py](/berserk_model_search.py). On the plus side, the WDL model described above is certainly the most important when it comes to the generated policy recommendations below, so there is not quite so much riding on the veridicality and accuracy of this berserk model and the next model I will describe. 

Even so, we can see in this plot that the model does a tolerable job. The model predicted opponent berserk probability is the orange curve, and the blue dots are the empirical relative frequencies from the data. I didn't bother to include any measure of confidence for the empirical values in this plot. We can see the model generally does OK, except for some minor troubles at the lower end of the ratings. 

![3_berserk_probs.gif](/plots/3_berserk_probs.gif)

### Modeling the probability of your opponent's rating
We also need a model of how likely you are to be matched with a player of a given rating. Again, some relevant factors here I neglected to record when processing the PGNs, so we resort again to a quick and dirty model. In this case, I consider only your own rating as a predictor variable for your opponent's. The reasoning is that arena tournament pairing works by attempting to pair you with someone close to you in terms of current tournament ranking. In the limit, all else being equal, tournament ranking should equate to relative player rating.

I first split player ratings into bins of width 100 rating points, then modeled the distributions of opponent ratings within each of these as beta distributions. Then, I fit smoothing splines to the progressions over the two beta distribution parameters (a, b) for each player rating bin. The plot below shows the data together with the model. The x-axis in each subplot is the opponent rating, while the title of each subplot is the player rating bin. We can see that the model fails to capture fully the spikes that occur in the empirical distributions for some bins, and over- or under-estimates the tails of some others. Still, for my quick-and-dirty purposes, it seems fine enough.

![3_opponent_rating_prob_model_vs_data.png](/plots/3_opponent_rating_prob_model_vs_data.png)

Now we can put all of these models together.

## Modeling arena tournament play as a Markov decision process
So far we only looked at whether to berserk when considering one game in isolation. But what we are really interested in is maximizing our points over the course of a whole tournament. To model this, we need to consider how our decision to berserk in one game affects our point-earning prospects in following games. For example, if you are currently on a streak, then losing a game means you'll lose the ability to earn double points on wins for at least 2 games. Even if the expected value of berserking on the currrent game (when considered in isolation) is higher than the expected value of not berserking, this might be outweighed by the potential loss of points in future games, so that the best decision is in fact not to berserk. How do we model this? 

It turns out that playing in an arena tournament can essentially be viewed as a [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP). The figure below demonstrates a simplified model. You are always in one of 3 states corresponding to your current win streak: the nodes 0, 1, or 2+. There are two possible actions in each state: to berserk (red lines/nodes) or not (blue lines/nodes). And for each state-action pair, there are two possible results: win or lose (assume for simplicity that draws aren't possible in this example). Each of these possible results has a reward associated with it: the points you earn (green numbers). Finally, each transition to a new state given the current state-action pair has some (possibly 0) probability of occurring (only the transitions with non-zero probability are depicted in this example).

![simple MDP for arena tournament play without draws](/images/simple_mdp_no_draws.png "simple MDP for arena tournament play without draws")

Once you have formulated a given problem as an MDP, there are standard techniques to determine the optimal policy that maximizes the total reward over time. In our case, we just need to go beyond this toy model by adding some more dimensionality to the state space. Specifically, we consider that each possible tuple of `(your win streak, your color, opponent rating, opponent berserked)` is a state. (To balance between keeping the problem discrete and not having an unmanageable combinatorial explosion in the number of states, we quantize ratings to bins of width 10.) Also, of course, we include the possibility of draws. All that remains is to determine the probability of each state-action-state transition. This is where the models we created in the above sections come in. The WDL model predicts, given your current state, what your streak will be next game. The opponent berserk model predicts whether your opponent will berserk in the next game. The opponent rating model predicts what your opponent's rating will be in the next game. Finally, we assume you are equally likely to have white or black in your next game. 

In this setup I have cobbled together, I have made various assumptions of varying questionability so as to make the modeling tractable, or to make the best of the data I had:
- I assume a tournament of infinite duration. In actuality, standard bullet tournaments are 30 minutes, blitz are 1 hour, Titled Arenas are 2 hours, and Blitz Titled Arenas are 3 hours. 
- I assume all games take the same amount of time. In particular, I assume games in which neither side berserked take no longer than games in which one or both sides berserked. It would have been possible to record total game time for each game when processing the PGNs, but I didn't think to do so at the time. However, a casual inspection of the total games played by competitors in a handful of Titled Arenas and Blitz Titled Arenas showed that this assumption is perhaps not so dreadfully inaccurate as it might initially seem.
- I assume the tournament has an infinite number of players. This is so that the assumptions behind the model predicting your opponent's rating work out.
- I assume no other competitor is learning or has learned an optimal policy; they all are assumed to act according the models fit to the data above.

Had I had these ideas clearly in mind when I initially set out to collect this data, I would have removed these assumptions, and considered more detailed, continuous finite-time MDPs where the tournament duration, competitors' rating distribution, competitors' moment-to-moment rankings/points, competitors' individual strategies, and other factors all come into play. In that case, multi-agent reinforcement learning and (evolutionary) game theory techniques could be applied, which would provide a much richer and more accurate answer to this overall question. 

I did not do that though, so in conclusion, here are the results for what I did do:

### MDP optimal policy for berserking vs. not berserking
We recapitulate the policy plots from earlier, now using the optimal policy found for the MDP. 

![3_berserk_better_mdp.png](/plots/3_berserk_better_mdp.png)

### MDP values for berserking vs. not berserking
Similarly, we can look at the value of berserking vs. not berserking, as calculated in the solution to the MDP. Note that these values aren't exactly the same as the expected values in earlier plots. These values don't directly correspond tournament points, since they depend on the temporal discounting parameter used in solving the MDP. 

![3_value_difference_mdp.png](/plots/3_value_difference_mdp.png)