# 🧠 Nim AI with Reinforcement Learning

This project uses **Q-learning**, a reinforcement learning algorithm, to teach an AI how to play the game of **Nim**. The AI learns by playing thousands of games against itself and improving its strategy over time.

## 🎮 What is Nim?

Nim is a mathematical game of strategy where players take turns removing objects from piles. On each turn, a player must remove **at least one object** from a **single pile**. The player who removes the **last object loses**.

## 🤖 How the AI Learns

- **States** are represented as a list of integers (e.g., `[1, 3, 5, 7]`) describing how many objects remain in each pile.
- **Actions** are tuples `(pile, count)` like `(2, 1)`, meaning "remove 1 object from pile 2".
- **Q-values** represent how good an action is in a given state.

### Q-Learning Formula:
```
Q(s, a) ← Q(s, a) + α * ((reward + max Q(s’, a’)) − Q(s, a))
```

- `s`: current state  
- `a`: action taken  
- `s’`: resulting state  
- `a’`: future possible actions  
- `α`: learning rate

The AI chooses actions using an **epsilon-greedy policy**, which means it sometimes explores new actions and sometimes exploits the best-known action.

## 🛠 Files

- `nim.py`: Game logic + Q-learning implementation
- `play.py`: Trains the AI and lets you play against it

## 🚀 Run It

```bash
python play.py
```

You’ll see training progress like:
```
Playing training game 1
...
Playing training game 10000
Done training

Piles:
Pile 0: 1
Pile 1: 3
Pile 2: 5
Pile 3: 7

AI's Turn
AI chose to take 1 from pile 2.
```

## 📚 What I Learned

- Implemented Q-learning from scratch
- Modeled turn-based strategy games with reinforcement learning
- Balanced exploration and exploitation with epsilon-greedy policy

---

👤 **[Your Name]**  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
