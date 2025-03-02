from collections import deque
from flask import Flask, render_template, jsonify, request
import math
import numpy as np
import os
import random
import sys
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import webbrowser

BLACK = 1
WHITE = -1
EMPTY = 0
BLACK_EMOJI = "⚫"
WHITE_EMOJI = "⚪"
EMPTY_EMOJI = "🟩"


class ReversiNN(nn.Module):
    def __init__(self, board_size):
        super(ReversiNN, self).__init__()
        self.board_size = board_size
        input_channels = 3
        conv_channels = 32 if board_size <= 8 else 64
        fc_size = 64 if board_size <= 8 else 128
        self.conv1 = nn.Conv2d(input_channels, conv_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels)
        self.policy_conv = nn.Conv2d(conv_channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_conv = nn.Conv2d(conv_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, fc_size)
        self.value_fc2 = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value


class ReversiState:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = BLACK
        center = board_size // 2
        self.board[center - 1][center - 1] = WHITE
        self.board[center][center] = WHITE
        self.board[center - 1][center] = BLACK
        self.board[center][center - 1] = BLACK
        self._valid_moves = None

    def clone(self):
        clone = ReversiState(self.board_size)
        clone.board = np.copy(self.board)
        clone.current_player = self.current_player
        clone._valid_moves = None
        return clone

    def get_valid_moves(self):
        if self._valid_moves is None:
            self._valid_moves = self._calculate_valid_moves()
        return self._valid_moves

    def _calculate_valid_moves(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves

    def is_valid_move(self, row, col):
        if self.board[row][col] != EMPTY:
            return False
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r][c] == -self.current_player
            ):
                pieces_to_flip.append((r, c))
                r += dr
                c += dc
            if (
                pieces_to_flip
                and 0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r][c] == self.current_player
            ):
                return True
        return False

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row}, {col})")
        new_state = self.clone()
        new_state.board[row][col] = self.current_player
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and new_state.board[r][c] == -self.current_player
            ):
                pieces_to_flip.append((r, c))
                r += dr
                c += dc
            if (
                pieces_to_flip
                and 0 <= r < self.board_size
                and 0 <= c < self.board_size
                and new_state.board[r][c] == self.current_player
            ):
                for fr, fc in pieces_to_flip:
                    new_state.board[fr][fc] = self.current_player
        new_state.current_player = -self.current_player
        new_state._valid_moves = new_state._calculate_valid_moves()
        if not new_state._valid_moves:
            new_state.current_player = -new_state.current_player
            new_state._valid_moves = new_state._calculate_valid_moves()
        return new_state

    def is_game_over(self):
        if not self.get_valid_moves():
            original_player = self.current_player
            self.current_player = -self.current_player
            opponent_has_moves = bool(self._calculate_valid_moves())
            self.current_player = original_player
            return not opponent_has_moves
        return False

    def get_winner(self):
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return EMPTY

    def get_score(self):
        if not self.is_game_over():
            return 0
        winner = self.get_winner()
        if winner == EMPTY:
            return 0
        elif winner == self.current_player:
            return 1
        else:
            return -1

    def display(self):
        print("  ", end="")
        for j in range(self.board_size):
            print(f"{j}", end=" ")
        print()
        for i in range(self.board_size):
            print(f"{i} ", end="")
            for j in range(self.board_size):
                if self.board[i][j] == BLACK:
                    print(f"{BLACK_EMOJI}", end="")
                elif self.board[i][j] == WHITE:
                    print(f"{WHITE_EMOJI}", end="")
                else:
                    print(f"{EMPTY_EMOJI}", end="")
            print()
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        print(f"Score - {BLACK_EMOJI}: {black_count}  {WHITE_EMOJI}: {white_count}")


class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    def expand(self, policy):
        valid_moves = self.state.get_valid_moves()
        for move in valid_moves:
            move_idx = move[0] * self.state.board_size + move[1]
            if move_idx < len(policy):
                prior = policy[move_idx]
                if move not in self.children:
                    try:
                        new_state = self.state.make_move(move[0], move[1])
                        self.children[move] = MCTSNode(
                            new_state, parent=self, move=move, prior=prior
                        )
                    except ValueError:
                        continue

    def select_child(self, c_puct=1.0):
        best_score = -float("inf")
        best_child = None
        best_move = None
        for move, child in self.children.items():
            if child.visits > 0:
                q_value = child.value_sum / child.visits
                ucb_score = q_value + c_puct * child.prior * math.sqrt(self.visits) / (
                    1 + child.visits
                )
            else:
                ucb_score = c_puct * child.prior * math.sqrt(self.visits + 1e-8)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                best_move = move
        return best_move, best_child

    def update(self, value):
        self.visits += 1
        self.value_sum += value

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, model, num_simulations=100, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, state):
        root = MCTSNode(state)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            while not node.is_leaf() and not node.state.is_game_over():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            if not node.state.is_game_over():
                policy, value = self._predict(node.state)
                node.expand(policy)
            else:
                value = node.state.get_score()
            for node in reversed(search_path):
                node.update(-value)
                value = -value
        moves = []
        visit_counts = []
        for move, child in root.children.items():
            moves.append(move)
            visit_counts.append(child.visits)
        total_visits = sum(visit_counts) or 1
        visit_probs = [count / total_visits for count in visit_counts]
        return list(zip(moves, visit_probs))

    def _predict(self, state):
        x = self._prepare_input(state)
        with torch.no_grad():
            policy_logits, value = self.model(x)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()
        return policy, value

    def _prepare_input(self, state):
        board = state.board
        current_player = state.current_player
        channel1 = np.zeros((state.board_size, state.board_size), dtype=np.float32)
        channel2 = np.zeros((state.board_size, state.board_size), dtype=np.float32)
        channel3 = np.zeros((state.board_size, state.board_size), dtype=np.float32)
        channel1[board == current_player] = 1
        channel2[board == -current_player] = 1
        channel3[board == EMPTY] = 1
        x = np.stack([channel1, channel2, channel3])
        x = torch.FloatTensor(x).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        return x


class SelfPlayTrainer:
    def __init__(
        self, board_size=8, mcts_simulations=100, batch_size=32, buffer_size=10000
    ):
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.model_black = ReversiNN(board_size)
        self.model_white = ReversiNN(board_size)
        if torch.cuda.is_available():
            self.model_black = self.model_black.cuda()
            self.model_white = self.model_white.cuda()
            print("Using GPU for training")
        else:
            print("Using CPU for training")
        self.optimizer_black = optim.Adam(self.model_black.parameters(), lr=0.001)
        self.optimizer_white = optim.Adam(self.model_white.parameters(), lr=0.001)
        self.buffer_black = deque(maxlen=buffer_size)
        self.buffer_white = deque(maxlen=buffer_size)
        self.game_counter = 0
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(f"model_black_{self.board_size}.safetensor"):
                self.model_black.load_state_dict(
                    torch.load(f"model_black_{self.board_size}.safetensor")
                )
                print(
                    f"Loaded black model for {self.board_size}x{self.board_size} board"
                )
            if os.path.exists(f"model_white_{self.board_size}.safetensor"):
                self.model_white.load_state_dict(
                    torch.load(f"model_white_{self.board_size}.safetensor")
                )
                print(
                    f"Loaded white model for {self.board_size}x{self.board_size} board"
                )
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting with fresh models")

    def save_models(self):
        try:
            torch.save(
                self.model_black.state_dict(), f"model_black_{self.board_size}.safetensor"
            )
            torch.save(
                self.model_white.state_dict(), f"model_white_{self.board_size}.safetensor"
            )
            print(f"Saved models for {self.board_size}x{self.board_size} board")
        except Exception as e:
            print(f"Error saving models: {e}")

    def self_play_game(self):
        state = ReversiState(self.board_size)
        game_history = []
        mcts_black = MCTS(self.model_black, num_simulations=self.mcts_simulations)
        mcts_white = MCTS(self.model_white, num_simulations=self.mcts_simulations)
        while not state.is_game_over():
            current_player = state.current_player
            if current_player == BLACK:
                mcts = mcts_black
            else:
                mcts = mcts_white
            action_probs = mcts.search(state)
            state_tensor = mcts._prepare_input(state)
            game_history.append((state_tensor, action_probs, current_player))
            if action_probs:
                moves, probs = zip(*action_probs)
                move_idx = np.random.choice(len(moves), p=probs)
                move = moves[move_idx]
                state = state.make_move(move[0], move[1])
            else:
                state.current_player = -state.current_player
                state._valid_moves = state._calculate_valid_moves()
        winner = state.get_winner()
        for state_tensor, action_probs, player in game_history:
            policy = torch.zeros(self.board_size * self.board_size)
            for move, prob in action_probs:
                move_idx = move[0] * self.board_size + move[1]
                policy[move_idx] = prob
            if winner == EMPTY:
                reward = 0.0
            elif winner == player:
                reward = 1.0
            else:
                reward = -1.0
            if player == BLACK:
                self.buffer_black.append((state_tensor, policy, reward))
            else:
                self.buffer_white.append((state_tensor, policy, reward))
        self.game_counter += 1
        if self.game_counter % 10 == 0:
            self.save_models()
        return winner

    def train(self):
        loss_black = self.train_model(
            self.model_black, self.optimizer_black, self.buffer_black
        )
        loss_white = self.train_model(
            self.model_white, self.optimizer_white, self.buffer_white
        )
        return loss_black, loss_white

    def train_model(self, model, optimizer, buffer):
        if len(buffer) < self.batch_size:
            return 0.0
        indices = random.sample(range(len(buffer)), self.batch_size)
        states, policies, rewards = [], [], []
        for idx in indices:
            state, policy, reward = buffer[idx]
            states.append(state)
            policies.append(policy)
            rewards.append(reward)
        state_batch = torch.cat(states)
        policy_batch = torch.stack(policies)
        reward_batch = torch.FloatTensor(rewards).view(-1, 1)
        if torch.cuda.is_available():
            policy_batch = policy_batch.cuda()
            reward_batch = reward_batch.cuda()
        policy_logits, value = model(state_batch)
        policy_loss = -torch.mean(
            torch.sum(policy_batch * F.log_softmax(policy_logits, dim=1), dim=1)
        )
        value_loss = F.mse_loss(value, reward_batch)
        total_loss = policy_loss + value_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return total_loss.item()


class ReversiGame:
    def __init__(self):
        self.board_size = 8
        self.state = None
        self.trainer = None
        self.black_player = "Human"
        self.white_player = "Human"
        self.training_thread = None
        self.stop_training = False

    def setup_game(self, board_size, black_player, white_player):
        self.board_size = board_size
        self.black_player = black_player
        self.white_player = white_player
        self.state = ReversiState(board_size)
        if self.trainer is None or self.trainer.board_size != board_size:
            self.trainer = SelfPlayTrainer(board_size, mcts_simulations=100)

    def get_ai_move(self):
        if self.state.current_player == BLACK:
            mcts = MCTS(self.trainer.model_black, num_simulations=100)
        else:
            mcts = MCTS(self.trainer.model_white, num_simulations=100)
        action_probs = mcts.search(self.state)
        if not action_probs:
            return None
        best_move, _ = max(action_probs, key=lambda x: x[1])
        return best_move

    def make_human_move(self, row, col):
        try:
            if (row, col) in self.state.get_valid_moves():
                self.state = self.state.make_move(row, col)
                return True
            return False
        except ValueError:
            return False

    def play_human_vs_human(self):
        self.state = ReversiState(self.board_size)
        while not self.state.is_game_over():
            self.state.display()
            current_player = "Black" if self.state.current_player == BLACK else "White"
            print(f"Current player: {current_player}")
            valid_moves = self.state.get_valid_moves()
            if not valid_moves:
                print(f"No valid moves for {current_player}. Turn passes.")
                self.state.current_player = -self.state.current_player
                self.state._valid_moves = self.state._calculate_valid_moves()
                continue
            print("Valid moves:", end=" ")
            for move in valid_moves:
                print(f"({move[0]},{move[1]})", end=" ")
            print()
            valid_input = False
            while not valid_input:
                try:
                    row = int(input("Enter row: "))
                    col = int(input("Enter column: "))
                    if self.make_human_move(row, col):
                        valid_input = True
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter numbers.")
        self.state.display()
        winner = self.state.get_winner()
        if winner == BLACK:
            print("Black wins!")
        elif winner == WHITE:
            print("White wins!")
        else:
            print("It's a draw!")

    def play_human_vs_ai(self):
        self.state = ReversiState(self.board_size)
        while not self.state.is_game_over():
            self.state.display()
            current_player = "Black" if self.state.current_player == BLACK else "White"
            player_type = (
                self.black_player
                if self.state.current_player == BLACK
                else self.white_player
            )
            print(f"Current player: {current_player} ({player_type})")
            valid_moves = self.state.get_valid_moves()
            if not valid_moves:
                print(f"No valid moves for {current_player}. Turn passes.")
                self.state.current_player = -self.state.current_player
                self.state._valid_moves = self.state._calculate_valid_moves()
                continue
            if player_type == "Human":
                print("Valid moves:", end=" ")
                for move in valid_moves:
                    print(f"({move[0]},{move[1]})", end=" ")
                print()
                valid_input = False
                while not valid_input:
                    try:
                        row = int(input("Enter row: "))
                        col = int(input("Enter column: "))
                        if self.make_human_move(row, col):
                            valid_input = True
                        else:
                            print("Invalid move. Try again.")
                    except ValueError:
                        print("Invalid input. Please enter numbers.")
            else:
                print("AI is thinking...")
                time.sleep(0.5)
                move = self.get_ai_move()
                if move:
                    self.state = self.state.make_move(move[0], move[1])
                    print(f"AI plays: ({move[0]}, {move[1]})")
                else:
                    print("AI has no valid moves. Turn passes.")
                    self.state.current_player = -self.state.current_player
                    self.state._valid_moves = self.state._calculate_valid_moves()
                time.sleep(0.5)
        self.state.display()
        winner = self.state.get_winner()
        if winner == BLACK:
            print("Black wins!")
        elif winner == WHITE:
            print("White wins!")
        else:
            print("It's a draw!")

    def play_ai_vs_ai_game(self):
        self.state = ReversiState(self.board_size)
        while not self.state.is_game_over():
            self.state.display()
            current_player = "Black" if self.state.current_player == BLACK else "White"
            print(f"Current player: {current_player} (AI)")
            print("AI is thinking...")
            time.sleep(0.3)
            move = self.get_ai_move()
            if move:
                self.state = self.state.make_move(move[0], move[1])
                print(f"AI plays: ({move[0]}, {move[1]})")
            else:
                print("AI has no valid moves. Turn passes.")
                self.state.current_player = -self.state.current_player
                self.state._valid_moves = self.state._calculate_valid_moves()
            time.sleep(0.3)
        self.state.display()
        winner = self.state.get_winner()
        if winner == BLACK:
            print("Black AI wins!")
        elif winner == WHITE:
            print("White AI wins!")
        else:
            print("It's a draw!")
        if not self.stop_training:
            time.sleep(1)
            self.play_ai_vs_ai_game()

    def start_training(self):
        self.stop_training = False
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        self.play_ai_vs_ai_game()

    def stop_training_thread(self):
        self.stop_training = True
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1.0)

    def training_loop(self):
        print("Starting AI training in background...")
        game_counter = 0
        while not self.stop_training:
            winner = self.trainer.self_play_game()
            game_counter += 1
            if game_counter % 5 == 0:
                loss_black, loss_white = self.trainer.train()
            if game_counter % 10 == 0:
                self.trainer.save_models()
                black_size = len(self.trainer.buffer_black)
                white_size = len(self.trainer.buffer_white)
                print(
                    f"Training progress: {game_counter} games played, buffer sizes: Black={black_size}, White={white_size}"
                )


app = Flask(__name__, static_folder="static", template_folder="templates")
game_instance = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def initialize_game():
    global game_instance
    data = request.get_json()
    board_size = int(data.get("boardSize", 8))
    black_player = data.get("blackPlayer", "Human")
    white_player = data.get("whitePlayer", "AI")
    game_instance.setup_game(board_size, black_player, white_player)
    if black_player == "AI":
        ai_move = game_instance.get_ai_move()
        if ai_move:
            game_instance.state = game_instance.state.make_move(ai_move[0], ai_move[1])
    return get_game_state()


@app.route("/api/move", methods=["POST"])
def make_move():
    global game_instance
    if not game_instance:
        return jsonify({"error": "Game not initialized"})
    data = request.get_json()
    row = int(data.get("row"))
    col = int(data.get("col"))
    valid_moves = game_instance.state.get_valid_moves()
    if (row, col) in valid_moves:
        game_instance.state = game_instance.state.make_move(row, col)
        if (
            game_instance.state.current_player == BLACK
            and game_instance.black_player == "AI"
        ):
            ai_move = game_instance.get_ai_move()
            if ai_move:
                game_instance.state = game_instance.state.make_move(
                    ai_move[0], ai_move[1]
                )
        elif (
            game_instance.state.current_player == WHITE
            and game_instance.white_player == "AI"
        ):
            ai_move = game_instance.get_ai_move()
            if ai_move:
                game_instance.state = game_instance.state.make_move(
                    ai_move[0], ai_move[1]
                )
        return get_game_state()
    return jsonify({"error": "Invalid move"})


@app.route("/api/ai-move", methods=["POST"])
def ai_move():
    global game_instance
    if not game_instance:
        return jsonify({"error": "Game not initialized"})
    current_player = game_instance.state.current_player
    player_type = (
        game_instance.black_player
        if current_player == BLACK
        else game_instance.white_player
    )
    if player_type != "AI":
        return jsonify({"error": "Not an AI's turn"})
    ai_move = game_instance.get_ai_move()
    if ai_move:
        game_instance.state = game_instance.state.make_move(ai_move[0], ai_move[1])
    return get_game_state()


@app.route("/api/state")
def get_game_state():
    global game_instance
    if not game_instance:
        return jsonify({"error": "Game not initialized"})
    board_list = game_instance.state.board.tolist()
    valid_moves = game_instance.state.get_valid_moves()
    game_over = game_instance.state.is_game_over()
    winner = game_instance.state.get_winner() if game_over else 0
    black_count = np.sum(game_instance.state.board == BLACK)
    white_count = np.sum(game_instance.state.board == WHITE)
    return jsonify(
        {
            "board": board_list,
            "currentPlayer": game_instance.state.current_player,
            "validMoves": valid_moves,
            "gameOver": game_over,
            "winner": winner,
            "blackScore": int(black_count),
            "whiteScore": int(white_count),
        }
    )


def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")


def start_gui(game):
    global game_instance
    game_instance = game
    threading.Thread(target=open_browser).start()
    app.run(debug=False)


def main():
    game = ReversiGame()
    use_gui = "--gui" in sys.argv
    if use_gui:
        start_gui(game)
    else:
        while True:
            print("\n===== Reversi Game =====")
            print("1. Human vs Human")
            print("2. Human vs AI")
            print("3. AI vs Human")
            print("4. AI vs AI (with training)")
            print(
                "5. Change Board Size (currently "
                + str(game.board_size)
                + "x"
                + str(game.board_size)
                + ")"
            )
            print("6. Exit")
            choice = input("Enter your choice: ")
            if choice == "1":
                game.setup_game(game.board_size, "Human", "Human")
                game.play_human_vs_human()
            elif choice == "2":
                game.setup_game(game.board_size, "Human", "AI")
                game.play_human_vs_ai()
            elif choice == "3":
                game.setup_game(game.board_size, "AI", "Human")
                game.play_human_vs_ai()
            elif choice == "4":
                game.setup_game(game.board_size, "AI", "AI")
                game.start_training()
            elif choice == "5":
                try:
                    size = int(input("Enter board size (4-16): "))
                    if 4 <= size <= 16 and size % 2 == 0:
                        game.board_size = size
                        print(f"Board size set to {size}x{size}")
                    else:
                        print(
                            "Invalid size. Please enter an even number between 4 and 16."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif choice == "6":
                game.stop_training_thread()
                print("Thanks for playing!")
                break
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
