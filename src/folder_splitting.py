import split_folders


if __name__ == '__main__':
    #Splits my classes into train, test, val folders.
    split_folders.ratio('/Users/mt/Galvanize/capstones/street_art_classifier/data/img_for_different_five', output="/Users/mt/Galvanize/capstones/street_art_classifier/data/train_test_split", seed=42, ratio=(.64, .16, .2))