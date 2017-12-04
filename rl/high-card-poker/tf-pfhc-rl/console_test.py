from player import Player
from pfhc_game import PreFlopHighCardGame


p1 = Player("A", 100)
p2 = Player("B", 100)
game = PreFlopHighCardGame(p1, p2)
while not game.over:
    print(str(game))

    player_to_act = game.next_player
    print("Next player " + player_to_act.name)

    count_ba = game.round_count  # round count before action

    action = input("Action {f,c,b}: ")
    if action == 'b':
        amount = int(input("Amount to bet: "))
        game.bet(amount=amount)
    elif action == 'c':
        game.call()
    else:
        game.fold()

    print("Action performed\n")

    if count_ba < game.round_count:
        print("Win this round: " + str(game.last_gain[0]) + " | " + str(game.last_gain[1]) + "\n\n")

print(game)