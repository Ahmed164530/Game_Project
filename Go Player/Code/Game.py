import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import numpy as np
import json
import random

BOARD_SIZES = [9, 13, 19]

KOMI = 6.5  

class GoGame:
    def __init__(self, board_size=19, heuristic_func=None):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  
        self.current_player = -1  
        self.player_color = -1  
        self.captured_stones = {1: 0, -1: 0}  
        self.heuristic_func = heuristic_func
        self.move_log = []   
    
    def is_valid_move(self, x, y):
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0

    def make_move(self, x, y):
        if self.is_valid_move(x, y):
            self.board[x, y] = self.current_player
            self.capture_stones(x, y)
            self.move_log.append((self.current_player, x, y))  # تسجيل الحركة  
            self.current_player *= -1  
            return True
        return False

    def capture_stones(self, x, y):
        """Captures stones that are surrounded by the opponent."""
        color = self.board[x, y]
        enemy_color = -color
        captured = []

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == enemy_color:
                group = self.find_group(nx, ny, enemy_color)
                if self.is_group_surrounded(group, enemy_color):
                    captured.extend(group)
        for (cx, cy) in captured:
            self.board[cx, cy] = 0
            self.captured_stones[enemy_color] += 1

    def find_group(self, x, y, color):
        """Finds all connected stones of the same color starting from (x, y)."""
        group = []
        visited = set()
        self.dfs(x, y, color, visited)
        group.extend(visited)
        return group

    def dfs(self, x, y, color, visited):
        """Depth-First Search to explore connected stones."""
        if (x, y) in visited:
            return
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return
        if self.board[x, y] != color:
            return
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.dfs(x + dx, y + dy, color, visited)

    def is_group_surrounded(self, group, color):
        """Checks if a group is surrounded by the opponent."""
        enemy_color = -color
        for (x, y) in group:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                    return False
        return True

    def is_game_over(self):
        return not np.any(self.board == 0)

    def get_score(self):
        """Calculates score based on captured stones and adds komi for white player."""
        white_score = self.captured_stones[-1] + KOMI
        black_score = self.captured_stones[1]
        return black_score, white_score

    def save_game(self, filename):
        data = {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "captured_stones": self.captured_stones
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_game(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.board = np.array(data["board"], dtype=int)
        self.current_player = data["current_player"]
        self.captured_stones = data["captured_stones"]

    def get_possible_moves(self):
        """Get all possible valid moves."""
        moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_move(x, y):
                    moves.append((x, y))
        return moves

    def ai_move(self):
        """AI move using the selected heuristic function."""
        possible_moves = self.get_possible_moves()
        if possible_moves:
            if self.heuristic_func:
                best_move = self.heuristic_func(self.board, self.current_player)
            else:
                best_move = random.choice(possible_moves)
            return best_move
        return None
    
    def heuristic_func_1(self, board, maximizing_player):
        """
          First heuristic function: Count the number of pieces for the player
          (maximizing_player = 1 for the maximizing player, -1 for the minimizing player).
        """
        opponent_piece = -1 if maximizing_player else 1
    
        player_count = 0
        opponent_count = 0

        for row in board:
            for cell in row:
                if cell == player_piece:
                    player_count += 1
                elif cell == opponent_piece:
                    opponent_count += 1

        return player_count - opponent_count
    
    def heuristic_func_2(self, board, maximizing_player):
        """
        Second heuristic function: Count the number of possible moves for the player.
        (maximizing_player = 1 for the maximizing player, -1 for the minimizing player).
        """
        player_piece = 1 if maximizing_player else -1
        opponent_piece = -1 if maximizing_player else 1


        player_moves = self.game.get_possible_moves_for_player(board, player_piece)
        opponent_moves = self.game.get_possible_moves_for_player(board, opponent_piece)


        return len(player_moves) - len(opponent_moves)

    def minimax(self, board, depth, maximizing_player):
        """Basic Minimax algorithm without pruning."""
        if depth == 0 or self.game.is_game_over(board):
            return self.evaluate_board(board)

        possible_moves = self.game.get_possible_moves(board)
        if maximizing_player:
            max_eval = -float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.game.make_move(board_copy, move, 1)  # 1 for maximizing player
                eval = self.minimax(board_copy, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.game.make_move(board_copy, move, -1)  # -1 for minimizing player
                eval = self.minimax(board_copy, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
        
    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with Alpha-Beta Pruning."""
        if depth == 0 or self.game.is_game_over(board):
            return self.evaluate_board(board)

        possible_moves = self.game.get_possible_moves(board)
        if maximizing_player:
            max_eval = -float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.game.make_move(board_copy, move, 1)  # 1 for maximizing player
                eval = self.minimax_alpha_beta(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.game.make_move(board_copy, move, -1)  # -1 for minimizing player
                eval = self.minimax_alpha_beta(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval
        
    def minimax_with_heuristic_1_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Minimax with Heuristic 1 and Alpha-Beta Pruning."""
        if depth == 0 or self.is_game_over():
            return self.heuristic_func_1(board, maximizing_player)

        possible_moves = self.get_possible_moves()
        if maximizing_player:
            max_eval = -float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.make_move(board_copy, move)
                eval = self.minimax_with_heuristic_1_alpha_beta(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.make_move(board_copy, move)
                eval = self.minimax_with_heuristic_1_alpha_beta(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval
        
    def minimax_with_heuristic_2_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Minimax with Heuristic 2 and Alpha-Beta Pruning."""
        if depth == 0 or self.is_game_over():
            return self.heuristic_func_2(board, maximizing_player)

        possible_moves = self.get_possible_moves()
        if maximizing_player:
            max_eval = -float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.make_move(board_copy, move)
                eval = self.minimax_with_heuristic_2_alpha_beta(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in possible_moves:
                board_copy = board.copy()
                self.make_move(board_copy, move)
                eval = self.minimax_with_heuristic_2_alpha_beta(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def pass_turn(self):
        
        self.move_log.append((self.current_player, 'pass', 'pass'))
        self.current_player *= -1   

class GoGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Go Player")
        self.board_size = 19
        self.bg_color = "#D4B483"
        self.black_stone_color = "black"
        self.white_stone_color = "white"
        self.ai_strategy = "Minimax"
        
        
        self.canvas = tk.Canvas(root, width=600, height=600, bg=self.bg_color)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click)
        
        self.create_score_display()
        self.setup_game()

    def create_score_display(self):
        self.score_label = tk.Label(self.root, text="White: 0 | Black: 0")
        self.score_label.pack()

    def setup_game(self):
        self.game = GoGame(self.board_size)
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        self.menu.add_command(label="New Game", command=self.new_game)
        self.menu.add_command(label="Save Game", command=self.save_game)
        self.menu.add_command(label="Load Game", command=self.load_game)
        self.menu.add_command(label="Change Board Size", command=self.change_board_size)
        self.menu.add_command(label="Select AI Strategy", command=self.select_ai_strategy)
        self.menu.add_command(label="Pass", command=self.pass_turn)
        self.menu.add_command(label="Resign", command=self.resign_game)
        self.menu.add_command(label="Quit", command=self.root.quit)
        self.draw_board()
    
    def select_ai_strategy(self):
        strategy = simpledialog.askstring(
            "AI Strategy",
            "Choose AI strategy:\n1. Minimax\n2. Minimax with Alpha-Beta\n3. Minimax with Heuristic 1\n4. Minimax with Heuristic 2"
        )
        strategy_map = {
            "1": "Minimax",
            "2": "Minimax with Alpha-Beta",
            "3": "Minimax with Heuristic 1",
            "4": "Minimax with Heuristic 2"
        }
        if strategy in strategy_map:
            self.ai_strategy = strategy_map[strategy]
            messagebox.showinfo("AI Strategy Selected", f"AI will now use: {self.ai_strategy}")
        else:
            messagebox.showerror("Invalid Choice", "Please choose a valid strategy.")

    def create_controls(self):
        self.score_label = tk.Label(self.root, text="White: 0 | Black: 0")
        self.score_label.pack()
        
        self.pass_button = tk.Button(self.root, text="Pass", command=self.pass_turn)
        self.pass_button.pack()

    def ai_move(self):
        move = None
        if self.ai_strategy == "Minimax":
            move = self.game.minimax(self.game.board, 3, True)
        elif self.ai_strategy == "Minimax with Alpha-Beta":
            move = self.game.minimax_alpha_beta(self.game.board, 3, -float('inf'), float('inf'), True)
        elif self.ai_strategy == "Minimax with Heuristic 1":
            move = self.game.minimax_with_heuristic_1_alpha_beta(self.game.board, 3, True)
        elif self.ai_strategy == "Minimax with Heuristic 2":
            move = self.game.minimax_with_heuristic_2_alpha_beta(self.game.board, 3, True)

        if move:
            x, y = move
            self.game.make_move(x, y)
            self.draw_board()
            if self.game.is_game_over():
                self.end_game()

    def evaluate_board(self, board):
        """The board evaluation based on captured stones and surrounded squares, with the addition of Komi points for the white player."""
        black_score = self.calculate_score(board, -1)  
        white_score = self.calculate_score(board, 1)   
    
        white_score += KOMI

        return black_score - white_score
    
    def calculate_score(self, board, player):
        """Calculating the score based on captured stones and surrounded squares."""
        captured_stones = self.game.captured_stones[player]
        enclosed_territory = 0
    
        
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] == 0 and self.is_enclosed(board, x, y, player):
                    enclosed_territory += 1

        return captured_stones + enclosed_territory
    
    def is_enclosed(self, board, x, y, player):
        """التحقق مما إذا كانت المربعات محاصرة بالكامل بواسطة لاعب معين."""
        visited = set()

        def dfs(x, y):
            if (x, y) in visited or x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
                return True
            if board[x, y] == -player:  
                  return False
            if board[x, y] == player:  
                  return True
        
            visited.add((x, y))
            return all(dfs(nx, ny) for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)])

        return dfs(x, y)

    def click(self, event):
        x, y = (event.y - 30) // 30, (event.x - 30) // 30
        if self.game.make_move(x, y):
            self.draw_board()
            if self.game.is_game_over():
                self.end_game()
            else:
                self.ai_turn()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.board_size):
            self.canvas.create_line(30, 30 + i * 30, 570, 30 + i * 30, fill="black")
            self.canvas.create_line(30 + i * 30, 30, 30 + i * 30, 570, fill="black")

        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.game.board[x, y] == 1:
                    self.canvas.create_oval(30 + y * 30 - 10, 30 + x * 30 - 10,
                                            30 + y * 30 + 10, 30 + x * 30 + 10, fill=self.white_stone_color)
                elif self.game.board[x, y] == -1:
                    self.canvas.create_oval(30 + y * 30 - 10, 30 + x * 30 - 10,
                                            30 + y * 30 + 10, 30 + x * 30 + 10, fill=self.black_stone_color)

        self.score_label.config(text=f"White: {self.game.captured_stones[-1]} | Black: {self.game.captured_stones[1]}")

    def ai_turn(self):
        move = self.game.ai_move()
        if move:
            x, y = move
            self.game.make_move(x, y)
            self.draw_board()
            if self.game.is_game_over():
                self.end_game()

    def end_game(self):
        black_score, white_score = self.game.get_score()
        winner = "Black" if black_score > white_score else "White"
        messagebox.showinfo("Game Over", f"Winner: {winner}\nBlack: {black_score} | White: {white_score}")
        self.root.quit()

    def save_game(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json")
        if filename:
            self.game.save_game(filename)

    def load_game(self):
        filename = filedialog.askopenfilename(defaultextension=".json")
        if filename:
            self.game.load_game(filename)
            self.draw_board()

    def new_game(self):
        self.setup_game()

    def change_board_size(self):
        size = simpledialog.askinteger("Board Size", "Enter board size (9, 13, or 19):")
        if size in BOARD_SIZES:
            self.board_size = size
            self.setup_game()
        else:
            messagebox.showerror("Invalid Size", "Invalid size. Please choose from 9, 13, or 19.")

    def resign_game(self):
        winner = "Black" if self.game.current_player == 1 else "White"
        messagebox.showinfo("Resignation", f"{winner} wins by resignation.")
        self.root.quit()

    def pass_turn(self):
        self.game.pass_turn()
        self.draw_board()
        self.ai_turn()

if __name__ == "__main__":
    root = tk.Tk()
    app = GoGameGUI(root)
    root.mainloop()
