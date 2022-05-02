The code is split up into 3 sections
    - board_handling -> feature detection method of identifying the board and constructing a homography
        - Main File: feature_detection.py (has the find_board function)
    - board_edge_detection -> edge detection method of identifying the board and constructing a homography
        - Main File: full_board_pipeline.py
    - train_detection -> classes and functions used to locate and check each connection on the board
        - Main File: Map.py

How to run
    1, `pip install -r requirements.txt`
    2, Choose an asset to assess in TTRScanner.py
    3, `py TTRScanner.py`