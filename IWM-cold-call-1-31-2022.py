from datetime import datetime, date
from Option740 import Option740
from Run import Run

today = date.today()

today = today.strftime("%Y%m%d")


r = Run('IWM', 196, 216, '20220304', '20220131', 2)
