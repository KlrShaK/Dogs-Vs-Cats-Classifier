# Dogs-Vs-Cats-Classifier
An ML Project to Classify the images of Dogs and Cats using Tensorflow.
The dataset on which the program is trained on is available here: https://www.kaggle.com/c/dogs-vs-cats

The Model have a 90% Validation Acccuracy on the dataset of 25,000 images
All Models were first trained on a small subset of the full dataset consisting of 2,500 images, for tuneing model parameters and achieveing maximum possible accuracy.

There are Multiple Versions with increasing performance and complexity.

* **V1(Un-named)** --> uses a custom written cnn with conv-normalize-pool architecture for extracting better features and to prevent problem of exploding gradients. It also uses -early stopping to prevent overfitting.
* **V2** --> Some minor updates to the architecture, improved user-interface. Feature maps of the input image are now shown to user to better elaborate the concept on CNNs to user.
* **V3** --> Uses InceptionV3 pre-trained on ImageNet as its base and with the help of transfer learning performance is improved


### Sample Feature Map
![myplot](https://user-images.githubusercontent.com/63470280/129925226-0bbc0ffe-41ab-4222-bd14-52ace4b08181.png)
