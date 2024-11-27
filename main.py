from state import State, State_2
import time
from importlib import import_module

x_wins = 0 
o_wins = 0 
draws_x = 0
draws_o = 0
times_out_x = 0
times_out_y = 0
times_elapsed_x = 0
times_elapsed_y = 0
def main(player_X, player_O, rule = 2):
    global x_wins, o_wins, draws_x, draws_o, times_out_x, times_out_y, times_elapsed_x, times_elapsed_y
    flag = False
    dict_player = {1: 'X', -1: 'O'}
    if rule == 1:
        cur_state = State()
    else:
        cur_state = State_2()
    turn = 1    

    limit = 81
    remain_time_X = 120
    remain_time_O = 120
    
    player_1 = import_module(player_X)
    player_2 = import_module(player_O)
    
    
    while turn <= limit:
        print("turn:", turn, end='\n\n')
        if cur_state.game_over:
            if cur_state.game_result == 0.:
                flag = True
                print("Draw")
                break
            print("winner:", dict_player[cur_state.player_to_move * -1])
            if dict_player[cur_state.player_to_move * -1] == "X":
                x_wins += 1
            elif dict_player[cur_state.player_to_move * -1] == "O":
                o_wins += 1
            break
        
        if len(cur_state.get_valid_moves) == 0:
            flag = True
            print("Draw")
            break
    
        start_time = time.time()
        if cur_state.player_to_move == 1:
            new_move = player_1.select_move(cur_state, remain_time_X)
            elapsed_time = time.time() - start_time
            remain_time_X -= elapsed_time
        else:
            new_move = player_2.select_move(cur_state, remain_time_O)
            elapsed_time = time.time() - start_time
            remain_time_O -= elapsed_time
            
        if new_move == None:
            break
        
        if remain_time_X < 0 or remain_time_O < 0:
            print("out of time")
            print("winner:", dict_player[cur_state.player_to_move * -1])
            if (cur_state.player_to_move == 1):
                times_out_x += 1
            else:
                times_out_y += 1
            break
                
        if elapsed_time > 10.0:
            print("elapsed time:", elapsed_time)
            print("winner: ", dict_player[cur_state.player_to_move * -1])
            if (cur_state.player_to_move == 1):
                times_elapsed_x += 1
            else:
                times_elapsed_y += 1
            break
        
        cur_state.act_move(new_move)
        print(cur_state)
        
        turn += 1
        
    print("X:", cur_state.count_X)
    print("O:", cur_state.count_O)
    if flag:
        if cur_state.count_X >= cur_state.count_O: # If you play O, change > to >= and vice verse.
            draws_x += 1
        else:
            draws_o += 1


# main('random_agent', '_MSSV')
for _ in range(10):
    print("Game", _, "\b:")
    # main('random_agent', '_multiProcessing')
    # main('_multiProcessing', 'random_agent')
    # main('random_agent', '_MSSV')
    main('_MSSV', 'random_agent')
    # main('_multiProcessing', '_multiProcessing')
    # main('_MSSV', '_MSSV')
    # main('_multiProcessing', '_MSSV')
    # main('_MSSV', '_multiProcessing')
print(f"X win: {x_wins}") 
print(f"O win: {o_wins}") 
print(f"Draw X win: {draws_x}")
print(f"Draw O win: {draws_o}")
print(f"X time out: {times_out_x}")
print(f"Y time out: {times_out_y}")
print(f"X time elapsed: {times_elapsed_x}")
print(f"Y time elapsed: {times_elapsed_y}")

 
