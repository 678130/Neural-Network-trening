Pitch Classification with CNN on Audio Spectrograms

The code in this project is from kaggle as we did not have cuda available to us. 
We did not run the training in python on local devices.
The training scripts do not run properly outside kaggle.
But all the code which is needed to understand and reproduce our results in kaggle have been provided

Kaggle notebook of final model:
https://www.kaggle.com/code/sondreh/dat255-prosjekt-cqt

Installation & Setup
1.
Create virtual environment
python -m venv venv

2.
Activate:
Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

3.
install dependencies
pip install -r requirements.txt

4.
Running the Demo

You should always run the demo with both:

Clean (original) audio - modeller/pitch_test.mp3
Augmented (noisy) version - modeller/pitch_test_augmented.mp3

MP3 files based on:
 Playback.fm. (2020). Perfect Pitch Test - Do You Have Absolute Pitch? [Video]. YouTube. https://www.youtube.com/watch?v=mFX94ZYHklg 
