from mahjong.mahjong_game import MahjongGame
from mahjong.player import Player


def main(num_simulations: int = 1):
    for i in range(num_simulations):
        print(f"Starting Simulation {i + 1}...")
        mahjong_game = MahjongGame(min_points_to_win=3, max_point_limit=13)

        while True:
            mahjong_game.play()

            next_wind = mahjong_game.determine_table_wind()

            if next_wind == mahjong_game.table_wind.EAST and mahjong_game.get_game_count() > 0:
                break

        # Print out all player scores.
        print(f"Simulation {i + 1} Results:")
        for idx, player in enumerate(mahjong_game.players, 1):
            print(f"Player {idx} score: {player.score}")
        print()


if __name__ == "__main__":
    main(1)
