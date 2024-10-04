import tkinter as tk
from tkinter import messagebox
import random


def check_winner(board, player):
    for row in board:
        if all([spot == player for spot in row]):
            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False


def check_draw(board):
    return all([spot != ' ' for row in board for spot in row])


def minimax(board, depth, alpha, beta, is_maximizing, difficulty):
    if check_winner(board, player_ai):
        return 1
    if check_winner(board, player_human):
        return -1
    if check_draw(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = player_ai
                    score = minimax(board, depth + 1, alpha, beta, False, difficulty)
                    board[i][j] = ' '
                    best_score = max(score, best_score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = player_human
                    score = minimax(board, depth + 1, alpha, beta, True, difficulty)
                    board[i][j] = ' '
                    best_score = min(score, best_score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
        return best_score


def ai_move(board, difficulty):
    if difficulty == "easy":

        empty_spots = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
        return random.choice(empty_spots)
    else:

        best_score = -float('inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = player_ai
                    score = minimax(board, 0, -float('inf'), float('inf'), False, difficulty)
                    board[i][j] = ' '
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move


def update_board(row, col, player, difficulty):
    if board[row][col] == ' ':
        board[row][col] = player
        buttons[row][col].config(text=player, state="disabled", disabledforeground="black")
        if check_winner(board, player):
            messagebox.showinfo("Game Over", f"{player} wins!")
            reset_board()
        elif check_draw(board):
            messagebox.showinfo("Game Over", "It's a draw!")
            reset_board()
        return True
    return False


def reset_board():
    global board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(text=" ", state="normal")

def user_click(row, col):
    if update_board(row, col, player_human, difficulty_level.get()):
        window.update()
        if difficulty_level.get() != "none":
            row, col = ai_move(board, difficulty_level.get())
            if row is not None and col is not None:
                update_board(row, col, player_ai, difficulty_level.get())


def choose_player_symbol():
    global player_human, player_ai
    player_human = symbol_var.get()
    player_ai = 'O' if player_human == 'X' else 'X'
    start_game()


def start_game():
    for widget in window.winfo_children():
        widget.destroy()

   
    tk.Label(window, text="Select Difficulty Level:", font=("Arial", 16), bg="lightblue").grid(row=0, column=0, columnspan=3)
    tk.Radiobutton(window, text="Easy", variable=difficulty_level, value="easy", font=("Arial", 14), bg="lightblue").grid(row=1, column=0)
    tk.Radiobutton(window, text="Medium", variable=difficulty_level, value="medium", font=("Arial", 14), bg="lightblue").grid(row=1, column=1)
    tk.Radiobutton(window, text="Hard", variable=difficulty_level, value="hard", font=("Arial", 14), bg="lightblue").grid(row=1, column=2)

    global board, buttons
    board = [[' ' for _ in range(3)] for _ in range(3)]
    buttons = [[None for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            buttons[i][j] = tk.Button(window, text=" ", font=('Arial', 40), width=5, height=2,
                                      command=lambda i=i, j=j: user_click(i, j),
                                      bg="white", fg="blue", relief="raised", borderwidth=5)
            buttons[i][j].grid(row=i+2, column=j, padx=10, pady=10)

    reset_button = tk.Button(window, text="Reset", font=('Arial', 20), command=reset_board, bg="lightgreen", fg="black")
    reset_button.grid(row=5, column=1, pady=20)


window = tk.Tk()
window.title("Tic-Tac-Toe")
window.configure(bg="lightblue")


difficulty_level = tk.StringVar(value="medium")
symbol_var = tk.StringVar(value="X")

tk.Label(window, text="Choose your symbol:", font=("Arial", 16), bg="lightblue").pack(pady=10)
tk.Radiobutton(window, text="X", variable=symbol_var, value="X", font=("Arial", 14), bg="lightblue").pack()
tk.Radiobutton(window, text="O", variable=symbol_var, value="O", font=("Arial", 14), bg="lightblue").pack()

start_button = tk.Button(window, text="Start Game", font=('Arial', 20), command=choose_player_symbol, bg="lightgreen", fg="black")
start_button.pack(pady=20)

window.mainloop()
