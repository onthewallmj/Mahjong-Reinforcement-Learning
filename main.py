from mahjong import Game


def main(num_simulations: int = 1):
    for i in range(num_simulations):
        print(f"Starting Simulation {i + 1}...")
        game = Game(min_points_to_win=3, max_point_limit=13)

        while True:
            game.play()

            # Check if we have completed a full cycle (returned to East wind)
            # The game logic updates table_wind internally during play() -> finalize_game() -> determine_table_wind() is not called explicitly there.
            # Actually, game.play() calls initialize_game() which calls determine_table_wind().
            # So the table wind is updated at the start of the game.

            # We need to check if the *next* game would start a new cycle.
            # game.determine_table_wind() calculates the wind for the *upcoming* game.
            current_table_wind = game.table_wind

            # If we are back at EAST wind and we have played some games (e.g. at least 4 hands, though realistically 16+ for a full game),
            # we can stop. But dealer retention makes game count unreliable.
            # Instead, we can rely on the fact that table_wind rotates E -> S -> W -> N -> E.

            # A simple heuristic for "Full Game" (4 rounds: East, South, West, North):
            # If the table wind is North, and the dealer passes from Player 3 to Player 0,
            # the wind will become East again.

            # Let's peek at what the wind *would* be for the next game.
            next_wind = game.determine_table_wind()

            # If the next wind is EAST, and we have history (meaning we've played at least one game),
            # and the previous game's wind was NORTH (to ensure we didn't just start or loop early),
            # then we are done.

            # To properly check "previous game's wind", we look at the last history entry.
            if game.history:
                last_game = game.history[-1]
                if next_wind == game.table_wind.EAST and last_game.table_wind == game.table_wind.NORTH:
                    # We just finished the North round.
                    break

                # Edge case: If the game ends exactly when North round ends.
                # But wait, determine_table_wind() updates self.table_wind.
                # So if we call it, self.table_wind becomes next_wind.

                if game.table_wind == game.table_wind.EAST and last_game.table_wind == game.table_wind.NORTH:
                    break

        # Print out all player scores.
        print(f"Simulation {i + 1} Results:")
        for idx, player in enumerate(game.players, 1):
            print(f"Player {idx} score: {player.score}")
        print()


if __name__ == "__main__":
    main(1)
