gst-launch-1.0 -vvv v4l2src device=/dev/video0 ! videoscale ! "video/x-raw,width=1920,height=1080" ! videoconvert ! "video/x-raw,format=RGB" ! filesink location=test.rgb24


gst-launch-1.0 -vvv v4l2src device=/dev/video0 ! videoscale ! "video/x-raw,width=1920,height=1080" ! videoconvert ! "video/x-raw,format=RGB" ! tcpclientsink host=192.168.1.239 port=9990
