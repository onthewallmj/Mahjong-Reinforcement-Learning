from mahjong_game import MahjongGame


def main():
    mahjong_game = MahjongGame(min_points_to_win=3, max_point_limit=13)
    while mahjong_game.get_game_count() < 20:
        mahjong_game.play()


if __name__ == "__main__":
    main()
