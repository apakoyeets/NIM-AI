import math
import random
import time


class Nim():
    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has:
          - piles: a list of how many objects remain in each pile
          - player: 0 or 1 to indicate whose turn it is
          - winner: None, 0, or 1 to indicate the winner (if there is one)
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Given a list of piles, return all available actions (i, j) where
        i is the index of a non-empty pile and j is the number of objects to remove.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Return the player who is not `player` (assumes player is 0 or 1).
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Change the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` (a tuple (i, j)) for the current player.
        """
        pile, count = action

        # Check for errors.
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update the pile.
        self.piles[pile] -= count
        self.switch_player()

        # If all piles are empty, the current player is declared the winner.
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize the AI.
          - self.q is a dictionary mapping (state, action) pairs to Q-values.
          - state: represented as a tuple of pile counts, e.g. (1, 3, 5, 7)
          - action: a tuple (i, j) representing taking j objects from pile i.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model given:
           old_state: the state before the move
           action: the move taken
           new_state: state after the move
           reward: the reward for taking this action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the (state, action) pair.
        If the pair has not been seen before, return 0.
        """
        # Since lists are unhashable, convert state to a tuple.
        state = tuple(state)
        return self.q.get((state, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for (state, action) using the formula:
        
            Q(s, a) <- old_q + alpha * ((reward + future_rewards) - old_q)
        """
        new_estimate = reward + future_rewards
        state = tuple(state)
        self.q[(state, action)] = old_q + self.alpha * (new_estimate - old_q)

    def best_future_reward(self, state):
        """
        Given a state, return the maximum Q-value among all possible actions in that state.
        If no actions exist, return 0.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        # For each possible action, if not present in self.q the Q-value is 0.
        return max(self.get_q_value(state, action) for action in actions)

    def choose_action(self, state, epsilon=True):
        """
        Given a state, return an action to take.
        
        If epsilon is True:
            With probability self.epsilon, choose a random available action.
            Otherwise, choose the action with the highest Q-value (breaking ties randomly).
        If epsilon is False:
            Always choose the best available action (i.e. greedily).
        """
        actions = list(Nim.available_actions(state))
        if not actions:
            return None

        # If epsilon is True, choose a random move with probability self.epsilon.
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        max_q = float("-inf")
        best_actions = []
        for action in actions:
            q_val = self.get_q_value(state, action)
            if q_val > max_q:
                max_q = q_val
                best_actions = [action]
            elif q_val == max_q:
                best_actions.append(action)
        return random.choice(best_actions)


def train(n):
    """
    Train an AI by having it play n games against itself.
    """
    player = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # For each player, keep track of the last state and action.
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Record the move for the current player.
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            # When the game is over, update Q-values with the final rewards.
            if game.winner is not None:
                # The move that led to game over receives a reward of -1.
                player.update(state, action, new_state, -1)
                # The previous move by the other player gets a reward of +1.
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # For intermediate moves, the reward is 0.
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return player


def play(ai, human_player=None):
    """
    Play a game of Nim against the trained AI.
    `human_player` can be set to 0 or 1 to have the human move first or second.
    """
    # If human_player is not set, randomly choose who goes first.
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return


if __name__ == "__main__":
    # Train the AI on 10,000 games.
    ai = train(10000)
    # Play a game against the human.
    play(ai)
