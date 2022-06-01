# NeRF-pytorch

To run NeRF-pytorch model:
```
bash scripts/bottles_baseline_61_noon.sh
```
Feel free to create new config files under `.config`, and modify the config file name correspondingly in script files under `.scripts`.

### Inference
set the below parameters in the config file
```
inference_mode = True # directly do inference without training
test_vids = [0, 16, 55, 93, 160] # pick the test view ids, as stated in problem 3.9 part 1 solution
```
