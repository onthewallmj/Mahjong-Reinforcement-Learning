from mahjong.game import Game
from mahjong.player import Player


def main(num_simulations: int = 1):
    for i in range(num_simulations):
        print(f"Starting Simulation {i + 1}...")
        game = Game(min_points_to_win=3, max_point_limit=13)

        while True:
            game.play()

            next_wind = game.determine_table_wind()

            if next_wind == game.table_wind.EAST and game.get_game_count() > 0:
                break

        # Print out all player scores.
        print(f"Simulation {i + 1} Results:")
        for idx, player in enumerate(game.players, 1):
            print(f"Player {idx} score: {player.score}")
        print()


if __name__ == "__main__":
    main(1)
