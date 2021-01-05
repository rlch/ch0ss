# ch0ss

```
          oooo          .oooo.
          `888         d8P'`Y8b
 .ooooo.   888 .oo.   888    888  .oooo.o  .oooo.o
d88' `"Y8  888P"Y88b  888    888 d88(  "8 d88(  "8
888        888   888  888    888 `"Y88b.  `"Y88b.
888   .o8  888   888  `88b  d88' o.  )88b o.  )88b
`Y8bod8P' o888o o888o  `Y8bd8P'  8""888P' 8""888P'
```

A sad excuse for a Chess AI, written to justify spending more time on my hopeless addiction

## Training Set

Trained on the [KingBase2019](https://archive.org/details/KingBase2019) dataset (2.2m+ games of players with 2000+ ELO). Aggregated into one file with this scriptkiddie nonsense:

```bash
[ ! -f dataset.pgn ] && cat *.pgn | sed '/\[.*\]/d' | tr -s '\n\r' | uniq > dataset.pgn || echo dont be greedy
```

## Value Function

The board state is valued by a fairly standard CNN. Feature extraction on the `n⨯6⨯8⨯8` design matrix is done through `Conv2D` layers, and classification through `Dense` layers to get a response in the range `[-1, 1]`. No pooling layers were used with the interest of retaining data - don't want the AI to end up with terrible board vision like me :upside_down_face: 

## Move Selection

With help from [python-chess](https://github.com/niklasf/python-chess) (getting valid moves + parsing PGNs), valid moves from a given position are searched using the minimax algorithm with alpha-beta pruning. 