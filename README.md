# Dog-Breed-Predictor

![Alt Image text](/flask_app_interface_example.png?raw=true "Flask application screenshot")

Dog-Breed-Predictor is a Python-3 Flask API that runs locally on your computer. 
Once started, you have to navigate to __localhost:5000__ (__127.0.0.1:5000__) with your web browser. 
A web interface will be displayed asking to upload a picture of a dog.
Once the picture is uploaded, a Convolutional Neural Network will predict the breed of the dog and the results will be displayed on the web page.

## More Information

The Convolutional Neural Network used here is the bottom part of the Xception model from the Keras library.
After a Global Average Pooling layer, a fully connected one has been added with 120 units (for the 120 dog breeds) and a softmax activation function. The network has first been trained on the Standford Dog Dataset, the base model being freezed. Then a fine tunning step, on the same dataset, has been performed with a low learning rate, un-freezing the last convolutional bloc of the base model. The network reached an accuracy of 0.83 on the test set.


## Installation

- To run the application, first download the entire repository. If you are cloning the repository with git, you will have to use git LFS (Large File Storage)  as one of the file is above 100 MB. In both cases, make sure that the _xception_120_breeds_fine_tuned.h5_ file is properly downloaded (around 120 MB).
- Once downloaded, it is recommended to create and activate a virtual environment using Python `venv`. 
On Windows, inside the repository you just downloaded, use the command:
    ```
    python -m venv env
    .\env\Scripts\activate
    ```
- Once the environment activated, you have to install the packages listed in _requirements.txt_. You can do so using `pip`:
    ```
    pip install -r requirements.txt
    ```
- You can finally run the application. Note that you may need admin rights to do so.
    ```
    python flask_app.py
    ```
- A local server will be launched on port __5000__ by default. Navigate to it with your browser.
You can then use the interface to upload picture and ask the model to determine the dog breed.

## Notes

All the submitted picture will be copied inside the _static\uploads_ folder, in the repository containing the Flask application. No cleaning is performed by the application once you closed it. You will have to manually delete the pictures to save disk space.

## Licence

This is a Free Software.
