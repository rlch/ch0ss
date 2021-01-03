# ch0ss

A sad excuse for a Chess AI, written to justify my hopeless addiction

## Training Set

Trained on the [KingBase2018](https://archive.org/details/KingBase2018) dataset (2m+ games of players with 2000+ ELO). Aggregated into one file with this scriptkiddie nonsense:

```bash
[ ! -f dataset.pgn ] && cat KingBase2018-06.pgn KingBase2018-05.pgn | sed '/\[.*\]/d' | tr -s '\n\r' | uniq > dataset.pgn || echo dont be greedy
```
