# blobtracking_python
a list of python examples on how to track blobs using different algorithms
Probably this code can be port to touchdesigner using Script top with Script chop and keep it all inside TD
Another solution is to send the data over osc or any other comunicaton protocol like websocket


# blobTrackingIRFeed.py
This is using an IR feed, few changes made from the other blob tracking. 
No background subtraction 
Directly isolate bright spots in the grayscale IR



# blobTrackingRealSenseIR.py 
 better to use a virtual env like conda #

 pip install pyrealsense2

 Like the other examples the threshold values has to be tweaked depending of the reflectivity of the IR emitter. 
 Better to add an UI system to tweak the threshold values

