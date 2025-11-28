from mahjong_game import MahjongGame
from player import Player


def main():
    mahjong_game = MahjongGame(min_points_to_win=3, max_point_limit=13)

    while True:
        mahjong_game.play()

        next_wind = mahjong_game.determine_table_wind()

        if next_wind == mahjong_game.table_wind.EAST and mahjong_game.get_game_count() > 0:
            break

    # Print out all player scores.
    for idx, player in enumerate[Player](mahjong_game.players, 1):
        print(f"Player {idx} score: {player.score}")


if __name__ == "__main__":
    main()
