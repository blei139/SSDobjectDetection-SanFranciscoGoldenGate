# SSDobjectDetection-SanFranciscoGoldenGate
This repository is a SSD object detection project, using the dash camera videoes taken from streets of the Golden Gate Bridge area in San Francisco. The object detection algorithm is SSD. The SSD model is 'ssd_mobilnet_v1_coco'. The object detection graph is 'frozen_inference_graph.pb', which has weights baked into the graph as constants and used by tensorflow for object detection. The ipython script, is SSDobjectDetectionVideoProc.ipynb. This script detects objects such stop signs, persons, cars, trucks, bicycles, and traffic lights. This script processes each frame in the input video by detecting objects, using openCV to detect traffic light color specifically for traffic lights. Then, it passes the processed image in each frame into a stream output video with SSD object detection and openCV traffic light color detection.

Instruction to run python script:
1) At the linux or ubuntu prompt, type this command in the object_detection directory:  python SSDobjectDetectionVideoProc.py  .

I have a very simple camera. My issue is that the road is super difficult to classify using my camera, but if I using the videos provided by Udacity's self-driving car curriculum, the road is painted well in most cases.
Youtube links using videos from Udacity's self-driving car curriculum: 

https://youtu.be/TFn7k3FSbVg

https://youtu.be/6TqYQ3kcgj0

https://youtu.be/o2pcyRc12Vs

https://youtu.be/xa-DvQLKOuI

https://youtu.be/TFn7k3FSbVg

See my youtube video here from my camera:

https://youtu.be/GBSllkG0u6o
