# street_art_classifier
![Header](images/my_images/cartoon_minds.JPG "Cartoon Minds")

## Table of Contents

1. [Description](#description)
2. [Repo Instructions](#repo)
3. [Data Sources](#ds)
4. [Data Processing](#dp)
5. [Results](#results)
6. [Summary](#summary)
7. [Next Steps](#next)
8. [Technologies Use](#tech)


<a name="description"></a>

# Web scraping
![Web_scraping](images/fat_cap_webpage.png "Fat Cap Webpage")

Data Overview

Wildstyle - 3131 | Walls only = 2656
Realistic - 2026 | Walls only = 1441

Cartoon - 2183 | Walls only = 1634
3D - 1015 | Walls only = 769
Brush - 442 | Walls only =243

# Images of each Style

---Wildstyle is a complicated and intricate form of graffiti. Due to its complexity, it is often very hard to read by people who are not familiar with it. Usually, this form of graffiti incorporates interwoven and overlapping letters and shapes. It may include arrows, spikes, and other decorative elements depending on the technique used. The numerous layers and shapes make this style extremely difficult to produce homogeneously, which is why developing an original style in this field is seen as one of the greatest artistic challenges to a graffiti writer. Wildstyle pieces are the most complex form of piece ("masterpiece") lettering, a stage higher than the quick simplified stylised letters known as "burners". Wildstyle is seen as one of the most complicated and difficult masterpiece styles and refers to larger complex letters which have volume as opposed to mere signatures or graffiti art "tags".--- Wikipedia

I decided to focus on Wildstyle and Realistic at the start both because they have the largest sample size out of the 18 styles, but also because they are very different from each other. Wildstyle is much more geometric and sharp lines focused on letters. Realistic is more portraits and landscapes. 
Data

## Inbalanced Classes
How to deal with inbalanced classes - I wrote a script to balance them through image augmentation. (Oversampling)

![Flow_Chart](images/balancing_function.png "Balancing Function")

## Model With Two Classes
150x150 
What the model sees vs what the actual image is.

### Confusion Matrix
![Confusion_Matrix](images/model_2/model_2_confusion_matrix.png "Confusion Matrix")

Images it got wrong

![Realistic_Label_Wildstyle_Prediction](images/model_2/piece_by_syde_-_orsay_(france)17486.jpg "Realistic labeled as Wildstyle")

Realistic: 0.0306063
Wildstyle: 0.99052274
Actual - Realistic
This was labeled realistic by whoever submitted to the website. While there are realistic parts of the image I feel like the Wildstyle section is most prominent. I would consider this a mislabel in the data.

../data/train_test_split/val/realistic/characters_by_carneiro_-_porto_(portugal)5391.jpg
Realistic: 0.42339113
Wildstyle: 0.51157093
Actual - Realistic
I'm not sure what it is picking up on this one to label it Wildstyle

'../data/train_test_split/val/wildstyle/characters_by_7same_-_bangkok_(thailand)6383.jpg'
Realistic: 0.93985313
Wildstyle: 0.2017667
Actual - Wildstyle
Again, I feel like this is a mislabeld

'../data/train_test_split/val/wildstyle/piece_by_kity_-_marseille_(france)12001.jpg'
Realistic: 0.7181519
Wildstyle: 0.12843975
Actual - Wildstyle
This one actually puzzels me. It appears to be a correct label in the original data, but I am not sure why the model classifies it as realistic

'../data/train_test_split/val/wildstyle/piece_by_kzed_-_amiens_(france)16487.jpg'

Realistic: 0.50048065
Wildstyle: 0.4919799
Actual - Wildstyle
Even mix. Not sure why

#Model With 5 Classes

Confusion Matrix
Images it got wrong
Which classes got confused together

#Running it on my own pictures