from mahjong import Game


def main(num_simulations: int = 1):
    for i in range(num_simulations):
        print(f"Starting Simulation {i + 1}...")
        game = Game(min_points_to_win=3, max_point_limit=13)

        while True:
            game.play()

            # Determine the next wind state.
            next_wind = game.determine_table_wind()

            # Check if we have completed a full cycle (returned to East wind from North).
            if game.history:
                last_game = game.history[-1]
                # Break if the next wind is East and we just finished a North round.
                if next_wind == game.table_wind.EAST and last_game.table_wind == game.table_wind.NORTH:
                    break

                # Also check if table_wind was already updated to East after a North round.
                if game.table_wind == game.table_wind.EAST and last_game.table_wind == game.table_wind.NORTH:
                    break

        # Print out all player scores.
        print(f"Simulation {i + 1} Results:")
        for idx, player in enumerate(game.players, 1):
            print(f"Player {idx} score: {player.score}")
        print()


if __name__ == "__main__":
    main(1)
