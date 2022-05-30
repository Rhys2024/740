from datetime import datetime, time
from Option740 import Option740


class Run():
    
    def __init__(self, ticker, price, exp, date_purchased, strike, contracts_purchased):
        
        self.ticker = ticker
        self.strike = strike 
        self.price = price
        self.exp = exp
        self.date_purchased = date_purchased
        self.n = contracts_purchased
        
        self.opt = Option740(self.ticker, self.price, self.exp, self.strike, self.date_purchased, self.n)
        
        self.run()
        
        
    def run(self):
        self.opt.live_updates()