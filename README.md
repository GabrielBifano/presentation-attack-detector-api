# **Face Presentation Attack Detector API**

This is a Machine Learning project 👨‍🎓🤖
Written with pytorch, and using a modified version of the CelebA Spoof dataset 🧮

The model achieved a **99.09%** overall accuracy 🔥. It works on JPG and JPEG only
( total correct over total predictions, so no precision vs accuracy taking place here 📊 )

## **You want to train it with some data?**
If you want to train the model you should be sure that your dataset folder has the following structure:

- 📂 root
  - 📁 live
  - 📁 spoof

'spoof' and 'live' dont need to be named spoof and live, and dont need to follow that specific order,
they must, however, exist (but if you switch the order, have in mind that the meaning of a positive 
result switches too). If you want to (and should) have a training, testing, and validating sets,
each of them must follow the dir structure presented, and loaded separately.
