import quantsbin.derivativepricing as qbdp
import yfinance as yf
from datetime import date, timedelta
from datetime import datetime as d
import datetime
from dateutil.relativedelta import *
import numpy as np
import yliveticker as live
import pandas as pd
import matplotlib.pyplot as plt

class Option740():
    
    def __init__(self, ticker, price, strike, date_purchased, exp, num_contracts):
        
        self.ticker = ticker
        self.exp = exp
        self.date_purchased = date_purchased
        self.strike = strike
        self.buy_price = price
        self.today = date.today()
        self.now = d.now()
        #self.current_price = live.YLiveTicker(on_ticker=on_new_msg, ticker_names=[ticker])
        self.volatility = self.get_vol()
        self.today_string = self.today.strftime("%Y%m%d")
        
        self.num_contracts = num_contracts
        
        self.option_module = qbdp.EqOption(option_type='Call', strike=self.strike, expiry_date = self.exp)

        self.option_at_purchase_date = self.option_module.engine(model='BSM', pricing_date=self.date_purchased, spot0= self.buy_price, rf_rate=0.02, volatility=self.volatility[-1])
        
        self.starting_greeks = self.option_at_purchase_date.risk_parameters()
        
        self.starting_valuation = self.option_at_purchase_date.valuation()
        
        #self.greeks_over_times = 
        
    def get_vol(self):
        
        closes = yf.download(self.ticker, start = self.today + timedelta(days = -120), end = self.today, progress=False)['Close']

        log_ret = np.log(closes/closes.shift(1))

        return log_ret.rolling(window=20).std() * np.sqrt(252)
    
    
    def live_updates(self):
        
        #self.live_profit = []
        
        #self.index = []
        
        live_prices = live.YLiveTicker(on_ticker=self.update, ticker_names=[self.ticker])
        
        #while (d.now().time() > datetime.time(hour=8, minute =0)) and (d.now().time() < datetime.time(hour=15, minute=0)):
            
            #self.update()
            
        print('Market is Closed')
        
    # , ws, msg    
    def update(self, ws, msg):
        
        self.current_price = msg['price']
        
        #self.current_price = yf.download(self.ticker, interval = '1m', start = d.now()+timedelta(minutes=-2), end = d.now(), progress = False)['Close'][-1]
        
        self.option_at_current_time = self.option_module.engine(model='BSM', pricing_date=self.today_string, spot0=self.current_price, rf_rate=0.02, volatility=self.volatility[-1])
        self.live_greeks = self.option_at_current_time.risk_parameters()
        
        self.live_delta_change = self.live_greeks['delta'] - self.starting_greeks['delta']
        
        #self.live_profit.append((self.option_at_current_time.valuation() - self.starting_valuation) * self.num_contracts * 100)
        #self.index.append(msg['timestamp'])
        
        self.live_profit = (self.option_at_current_time.valuation() - self.starting_valuation) * self.num_contracts * 100
        
        #plt.show()
        #plt.plot(self.index, self.live_profit)
        #plt.show(block=False)
        #plt.draw()
        #plt.pause(0.001)
        
        print(f'Profit: ${round(self.live_profit, 2)}')
        print(f'Current Price of {self.ticker}: ${round(self.current_price, 2)}\n\n')
        