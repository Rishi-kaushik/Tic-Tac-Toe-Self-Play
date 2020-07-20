# Tic-Tac-Toe Agent Training With Self-Play
## Overview
This project implements a **tic-tac-toe** training environment similar to openAI Env.<br>
It also trains a model via **self-play**.<br>
Env.render() is a simple CLI as self-play is the project focus.

## Results
The trained model performs well after about 100,000 games, but still falls for trapping strategies if it goes second.<br>
It is reliably win or draw is going first.

## File Navigation
- The environment code is in **TicTacToe_Env.py**<br>
- The code to train and play with the trained model is in **main.py**
- The trained model is stored in **models/..**

## Extra
- While playing with model the cell numbers are:<br>
1 2 3<br>
4 5 6<br>
7 8 9