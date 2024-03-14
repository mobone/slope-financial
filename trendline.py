import pandas as pd
from utils import fit_trendlines_high_low, get_line_points
import pandas as pd
import yfinance as yf


index_data = yf.Ticker('TQQQ').history(period='7d', interval='1m')
stock_data = yf.Ticker('NVDA').history(period='7d', interval='1m')
#stock_data = yf.Ticker('SMCI').history(period='7d', interval='1m')

# Take natural log of data to resolve price scaling issues
#index_data_log = np.log(index_data)
#stock_data_log = np.log(stock_data)

multiplier = 13.5

stock_data = stock_data / multiplier

lookback = 45

def plot_trendlines(data, i, symbol):
    candles = data.iloc[i - lookback + 1: i + 1]

    support_coefs, resist_coefs = fit_trendlines_high_low(candles['High'], candles['Low'], candles['Close'])

    return support_coefs, resist_coefs

all_results = []
for modifier in [.0,.005,.01,.015]:
    for other_modifier in [.0,.005,.01,.015]:
        holding = False
        results = []
        for i in range(lookback - 1, len(index_data)):
            
            index_support_coefs, index_resist_coefs = plot_trendlines(index_data, i,'TQQQ')
            
            stock_support_coefs, stock_resist_coefs = plot_trendlines(stock_data, i, 'NVDA')
            
            if holding == True and stock_support_coefs[0]>index_support_coefs[0]+other_modifier:
                this_data = stock_data.iloc[i - lookback + 1: i + 1]
                sell_price = round(float(this_data.tail(1)['Close'])*multiplier, 2)
                percent_change = round((sell_price-buy_price)/buy_price*100,1)
                sell_date = str(this_data.tail(1).index.values[0]).split('.')[0]

                #print('sold', buy_date, sell_date, buy_price, sell_price, str(percent_change)+'%')

                results.append([buy_date,sell_date,buy_price, sell_price, sell_price-buy_price, percent_change])
            
                holding = False

            if holding == False and index_support_coefs[0]>0 and index_support_coefs[0]>stock_support_coefs[0]+modifier:
                this_data = stock_data.iloc[i - lookback + 1: i + 1]
                buy_price = round(float(this_data.tail(1)['Close'])*multiplier,2)
                buy_date = str(this_data.tail(1).index.values[0]).split('.')[0]
                #print('buy', buy_date, buy_price)
                holding = True
            
            
            #plt.pause(0.0001)
            #plt.cla()
        
        result_df = pd.DataFrame(results, columns=['buy date', 'sell date', 'buy price', 'sell price', 'profit', 'percent_change'])
        print(result_df)

        if not result_df.empty:
            all_results.append([modifier, other_modifier, round(float(result_df['profit'].sum()),2), len(result_df[result_df['profit']>0]), len(result_df['profit']), len(result_df[result_df['profit']>0])/len(result_df['profit'])])
            print(pd.DataFrame(all_results, columns=['modifier', 'other modifier', 'return', 'profitable count', 'total count','profitable percent']))
            
            print()
        
        
        
