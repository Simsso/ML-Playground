from player import Player
from pfhc_game import PreFlopHighCardGame


p1 = Player("A", 100)
p2 = Player("B", 100)
game = PreFlopHighCardGame(p1, p2)
while not game.over:
    print(str(game))

    player_to_act = game.next_player
    print("Next player " + player_to_act.name)

    action = input("Action {f,c,b}: ")
    if action == 'b':
        amount = int(input("Amount to bet: "))
        game.bet(player_to_act, amount)
    elif action == 'c':
        game.call(player_to_act)
    else:
        game.fold(player_to_act)

    print("Action performed\n\n")

print(game)