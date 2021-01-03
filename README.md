# ch0ss

A sad excuse for a Chess AI, written to justify my hopeless addiction

## Training Set

Trained on the [KingBase2019](https://archive.org/details/KingBase2019) dataset (2.2m+ games of players with 2000+ ELO). Aggregated into one file with this scriptkiddie nonsense:

```bash
[ ! -f dataset.pgn ] && cat *.pgn | sed '/\[.*\]/d' | tr -s '\n\r' | uniq > dataset.pgn || echo dont be greedy
```


## Value Function

The board state is valued by a fairly standard CNN. Feature extraction on the `n⨯6⨯8⨯8` design matrix is done through `Conv2D` layers, and classification through `Dense` layers to get a response in the range `[-1, 1]`. No pooling layers were used with the interest of retaining data - don't want the AI to end up with terrible board vision like me :upside_down_face: 

## Move Selection

Powered by the handy [python-chess](https://github.com/niklasf/python-chess), valid moves from a given position are searched using the minimax algorithm with alpha-beta pruning. 