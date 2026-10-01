# cat_segment
A deep learning model for segmenting cats in photos. I am using data from oxford-iiit-pet and coco2017. coco2017 I only use the training data that include cats and nothing else. I perform a 80/20 split. I created a UNet although this UNet code is very un-optimized. I also plan to include attention blocks. The trained model was too large to upload, you can get the model from https://huggingface.co/jasonb-q/cat_segment.

Here is a little picture of some random climbing guy. bottom left I used the model to cut my cat, Belle, out of the original image and pasted her in with the climber. 
<br>
<img width="364" height="549" alt="climbbelle" src="https://github.com/user-attachments/assets/781b640c-e84d-4c7a-84be-3677cf996539" />
