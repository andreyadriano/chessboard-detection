# Chessboard detection
An approach using image processing and computer vision techniques

> :warning: The scripts were only tested on Ubuntu 24.04 LTS and should only work on Linux distributions.

## How does it work?

`TODO`

## How to setup environment

Create virtual environment:
```
python3 -m venv .venv
```

Activate virtual environment:
```
source .venv/bin/activate
```

Install dependencies:
```
pip install opencv-python
```

## How to run

After activating the python virtual environment, you can run the script using the following command:
```
python3 detect_chessboard.py <IMAGE_NAME> <OUTPUT_PATH>
```

You can use the `--show` option to show the image step-by-step:
```
python3 detect_chessboard.py <IMAGE_NAME> <OUTPUT_PATH> --show
```

### For multiple jpg images

If you want to run the script for many jpg images in a row, you can use the `run_detect_chessboard_for_all.sh` script, which will do it for all jpg images in an input path given as argument and save it in the output path.

> :warning: Keep in mind this script was written for jpg images only. If you have any other image extension, you will need to edit the script.

First, you need permission for running the script:
```
chmod +x run_detect_chessboard_for_all.sh
```

Then you can run it with:
```
./run_detect_chessboard_for_all.sh -i <INPUT_PATH> -o <OUTPUT_PATH>
```
For example:
```
./run_detect_chessboard_for_all.sh -i img -o output
```