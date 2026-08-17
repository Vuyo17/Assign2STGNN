| Model                     |   Epochs run | Early stopped   |   Total time (min) |   Avg s/epoch |   Best val MAE |
|:--------------------------|-------------:|:----------------|-------------------:|--------------:|---------------:|
| TTS                       |           30 | False           |              3.200 |         6.400 |          2.959 |
| GWN (predefined)          |           25 | True            |            329.600 |       791.100 |          2.685 |
| GWN (predefined+adaptive) |           15 | True            |            355.300 |      1421.100 |          2.623 |
| AGCRN                     |           21 | False           |            361.400 |      1032.700 |          2.708 |