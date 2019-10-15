import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio('/Users/mt/Galvanize/capstones/street_art_classifier-/data/img', output="/Users/mt/Galvanize/capstones/street_art_classifier-/data/train_test_split", seed=42, ratio=(.64, .16, .2))