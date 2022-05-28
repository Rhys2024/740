from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import fredapi as fa
import pandas as pd
import numpy as np
import os

def get_path(end_item):
    
    item_list = ['Users', os.getlogin(), 'Desktop', 'exposure_to_rates', end_item]
    
    path = "\\"
    
    for item in item_list:
        
        path = os.path.join(path, item)
        
    return path

def get_current_score(back, rate):
    
    rate = rate.dropna()

    grad_now = pd.Series([(grad-rate[num])/5 for num, grad in enumerate(rate[5:])], index = [i for i in rate[5:].index])

    grad_now_means = grad_now.rolling(252, center=False).mean()
    grad_now_std = grad_now.rolling(252, center=False).std()

    rate_means = rate.rolling(252, center=False).mean()
    rate_std = rate.rolling(252, center=False).std()

    g_now = (grad_now[-back:] - grad_now_means[-back:])/(grad_now_std[-back:])

    rate_now = (rate[-back:] - rate_means[-back:])/(rate_std[-back:])

    past_4_months_scores_yield_curve = round(g_now[-30:] + rate_now[-30:], 0).dropna()

    
    return int(round(past_4_months_scores_yield_curve.mean(),0))

def percent_success(df, similar_rates_or_pattern, reference_columns = None):
    
    pct_success = {o : 0.0 for o in df.drop(columns = reference_columns)}

    for fac in df.drop(columns = reference_columns):

        above_market = 0
        all_em = 0

        if isinstance(similar_rates_or_pattern, list):
            for i in similar_rates_or_pattern:

                if len(reference_columns) == 2:
                    ry_yc = df[fac].loc[ (df[reference_columns[0]] == i[0]) & (df[reference_columns[1]] == i[1])].to_list()
                    
                    above_market += len([r for r in ry_yc if r > 0.0])
                    all_em += len(ry_yc)
            
        else:
            ry_yc = df[fac].loc[ df[reference_columns[0]] == similar_rates_or_pattern].to_list()
            above_market += len([r for r in ry_yc if r > 0.0])
            all_em += len(ry_yc)
            
        pct_success[fac] = round(above_market/all_em ,2)

    return pct_success

def write_to_file(file):
    
    file.write(f"\nTimeStamp: {datetime.now()}")
    
    file.write("\n\n\nRATES DATA- Real Rates and 10 - 2: ")
    #if abs(current_real_rate) > 3:
        #file.write("\nRATES ARE BEING SHOCKED\n")
    file.write(f"\n\nCurrent Real Yield: {current_real_rate}")
    file.write(f"\nCurrent Yield Curve: {current_yield_curve}\n")
        
    file.write(f"\nMean Returns in Similar Rate Environments:\n")
    
    for num, ok in enumerate(combined_scores_means):
        
        file.write(f"{combined_scores_means.index[num]} : {round(ok,3)}\n")
        
    file.write("\n")
    file.write(f"Percentage of Similar Rate Signals Above Market:\n")
    
    for factor in percent_above_market:
        file.write(f"{factor} : {percent_above_market[factor]}\n")
    
    file.write("\n")
    
    file.write("95% Confidence Return Interval for Similar Rate Environments: \n")
    for num, (m, s) in enumerate(zip(combined_scores_means, combined_scores_std)):
        
        file.write(f"{combined_scores_means.index[num]} : {round(m - 3*s, 2),round(m + 3*s, 2)}\n")
        
    if num_signals >= 35:
        file.write(f"\nSignificance Level: STRONG ({num_signals} signals)") 
    else:
        file.write(f"\nSignificance Level: WEAK ({num_signals} signals)") 
    
    file.write(f"\n\n\n\nPATTERN DATA: \n")
    
    file.write(f"\nCurrent Yield Curve Pattern: {pattern_now}\n")
    
    file.write(f"\nMean Returns in {pattern_now} Patterns:\n")
    
    for num, ok in enumerate(mean_returns_per_pattern.T[pattern_now]):
        
        file.write(f"{mean_returns_per_pattern.T[pattern_now].index[num]} : {round(ok,3)}\n")
    
    file.write("\n")
    
    file.write(f"Percentage of {pattern_now} Signals Above Market:\n")
    
    for factor in factor_success_rate_for_current_pattern:
        
        if factor not in ['ry_scores', 'yc_scores']:
            file.write(f"{factor} : {factor_success_rate_for_current_pattern[factor]}\n")
    
    
    file.write(f"\n95% Confidence Return Interval for a {pattern_now} Yield Curve: \n")
    for num, (m, s) in enumerate(zip(mean_returns_per_pattern.T[pattern_now], std_returns_per_pattern.T[pattern_now])):
        
        file.write(f"{mean_returns_per_pattern.T[pattern_now].index[num]} : {round(m - 3*s, 2),round(m + 3*s, 2)}\n")
    
    file.write("\n\n")
    
    file.close()

def current():
    
    global today, forward_return_df
    global combined_scores_means, combined_scores_std, percent_above_market, num_signals
    global current_real_rate, current_yield_curve, path_factors

    today = date.today()
    start = today + timedelta(days = -252-215)
    back = 252
    
    fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')

    real_rate = fred.get_series('DFII10', observation_start = start, end = today)
    yield_curve = fred.get_series('T10Y2Y', observation_start = start, end = today)

    current_real_rate = get_current_score(back, real_rate)
    current_yield_curve = get_current_score(back, yield_curve)
    
    # Add the correct path for computer
    #path = os.path.realpath("/Users/rhys/Desktop/Heard-Cap/Factor_Project/factors_and_scores.xlsx")
    path_factors = get_path("factors_and_scores.xlsx")
    # print(f"\n\n{path}\n\n")
    #path = os.path.abspath("Factor_Project/factors_and_scores.xlsx")
    forward_return_df = pd.read_excel(path_factors, index_col="Date")
    
    similar_periods = []
    
    real_ranges = [s for s in range(current_real_rate-1, current_real_rate+2)]
    curve_ranges = [y for y in range(current_yield_curve-1, current_yield_curve+2)]
    
    num_signals = 0
    
    for i in real_ranges:
        for j in curve_ranges:
            similar_periods.append((i,j))
    
    combined_scores = pd.DataFrame()

    for rates in similar_periods:
        
        signals = forward_return_df[[j for j in forward_return_df.columns if "scores" not in j]].loc[
                                                                                                                    (forward_return_df.ry_scores == rates[0]) & 
                                                                                                                    (forward_return_df.yc_scores == rates[1])]

        if abs(current_real_rate - rates[0]) <= 1 and abs(current_yield_curve - rates[1]) <= 1:
            num_signals += len(pd.Series([datetime.strftime(d, "%Y-%m") for d in signals.index], dtype=object).unique())
        combined_scores[rates] = signals.mean()
    
    combined_scores = combined_scores.dropna(axis = 1).T
    combined_scores_means = round(combined_scores.mean(), 2)
    combined_scores_std = round(combined_scores.std(), 2)
    percent_above_market = percent_success(forward_return_df, similar_periods, ['ry_scores', 'yc_scores'])
    
    return current_real_rate

def get_patterns():
        
    ten_and_two = pd.DataFrame([ten_change_mean, two_change_mean]).T
    ten_and_two.columns = ['10-Year Change', '2-Year Change']

    ten_and_two['Bull Flattening'] = 0
    ten_and_two['Bear Flattening'] = 0
    ten_and_two['Bull Steepening'] = 0
    ten_and_two['Bear Steepening'] = 0
    #ten_and_two['Inverted Curve'] = 0
    ten_and_two['Pattern'] = 'None'

    for date in ten_and_two.index:
        
        if ten_and_two.loc[date, '10-Year Change'] > 0 and ten_and_two.loc[date, '2-Year Change'] > 0:
        
            if ten_and_two.loc[date, '10-Year Change'] > ten_and_two.loc[date, '2-Year Change']:
                
                ten_and_two.loc[date, 'Bear Steepening'] = 1
                ten_and_two.loc[date, 'Pattern'] = 'Bear Steepening'
            
            elif ten_and_two.loc[date, '10-Year Change'] < ten_and_two.loc[date, '2-Year Change']:
                
                ten_and_two.loc[date, 'Bear Flattening'] = 1
                ten_and_two.loc[date, 'Pattern'] = 'Bear Flattening'
                
        elif ten_and_two.loc[date, '10-Year Change'] < 0 and ten_and_two.loc[date, '2-Year Change'] < 0:
            
            if ten_and_two.loc[date, '10-Year Change'] > ten_and_two.loc[date, '2-Year Change']:
                
                ten_and_two.loc[date, 'Bull Steepening'] = 1
                ten_and_two.loc[date, 'Pattern'] = 'Bull Steepening'
            
            elif ten_and_two.loc[date, '10-Year Change'] < ten_and_two.loc[date, '2-Year Change']:
                
                ten_and_two.loc[date, 'Bull Flattening'] = 1
                ten_and_two.loc[date, 'Pattern'] = 'Bull Flattening'
    
    return ten_and_two


def current_pattern(forward = 252):
    
    
    global ten_change_mean, two_change_mean, patterns, pattern_df, df, lengths_df, yield_curve, pattern_now
    
    # Basic Allocation of Input Variables
    df = pd.DataFrame()
    today = date.today()
    start = today + timedelta(days = -252-5000)
    back = 252
    long = 'ten'
    short = 'two'
    
    fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')
    long_rate = fred.get_series('DGS10', observation_start = start, end = today)
    short_rate = fred.get_series('DGS2', observation_start = start, end = today)
    yield_curve = fred.get_series('T10Y2Y', observation_start = start, end = today)
    
    df[long] = long_rate.dropna()
    df[short] = short_rate.dropna()
    df['Yield Curve'] = yield_curve.dropna()
    forward = forward

    patterns = ['Bull Flattening', 'Bear Flattening', 'Bull Steepening', 'Bear Steepening']
    
    # Behavior of 10-Year vs 2-Year
    ten_change = pd.Series([(chng/df[long][num])-1 for num, chng in enumerate(df[long][30:])], index = [i for i in df[long][30:].index])
    two_change = pd.Series([(chng/df[short][num])-1 for num, chng in enumerate(df[short][30:])], index = [i for i in df[short][30:].index])
    ten_change_mean = ten_change.rolling(60).mean()[60:]
    two_change_mean = two_change.rolling(60).mean()[60:]

    pattern_df = get_patterns()
    lengths_df = get_pattern_lengths()
    df['Pattern'] = add_patterns()
    
    df['Inverted'] = is_inverted()
    df['Curve Behavior'] = combine()
    
    pattern_now = df['Curve Behavior'][-30:].value_counts().index[0]


def get_pattern_lengths():

    lengths = {pat : {} for pat in patterns}

    for p in patterns:

        for num, i in enumerate(pattern_df[p]):
            
            if num > 0 and i == 1 and pattern_df[p][num-1] != 1 and num + 1 < len(pattern_df):
                
                lengths[p][pattern_df.index[num]] = length_of_pattern(pattern_df[p][num:])
                
                
    return pd.DataFrame(lengths)


def add_patterns():

    count = 0
    general_pattern = []
    
    lengths_df['Yield Curve'] = df['Yield Curve']
    df['Types'] = pattern_df['Pattern'].fillna('None')
    df['Lengths'] = 0
    
    lengths = lengths_df.fillna(0.0)
    shortest_pattern = min([max(lengths[i]) for i in lengths if i != 'Yield Curve'])


    for date in lengths_df.index:
        
        for p in patterns:
            
                
            if lengths_df.loc[date, p] > (shortest_pattern - 10):
                
                df.loc[date, 'Lengths'] = lengths_df.loc[date, p]

    while count < len(df.index):
        
        if df.loc[df.index[count], 'Lengths'] > 0:
            
            for i in range(int(df.loc[df.index[count], 'Lengths'])):
                
                general_pattern.append(df.loc[df.index[count], 'Types'])
            
            count += int(df.loc[df.index[count], 'Lengths'])
            
        else:
            general_pattern.append('None')
                
            count += 1
            
    general_pattern = pd.Series(general_pattern, index = df.index)

    return general_pattern


def combine():
        
    combo = [(p,i) for p,i in zip(df['Pattern'], df['Inverted'])]

    pattern_inversion = []
    
    for pi in combo:
        
        if not pi[1]:
            pattern_inversion.append(pi[0])
        else:
            pattern_inversion.append('Inverted Curve')
            
    return pattern_inversion

def is_inverted():
    
    return np.select([df['Yield Curve'] >= 0.0, df['Yield Curve'] < 0.0], [0,1])
    

def length_of_pattern(series):

    count = 0
    
    while (series[count] != 0):
        
        count += 1
        
        if (count >= len(series)):
            return count
        
    return count

def get_pattern_means():
    
    global mean_returns_per_pattern, mean_returns_per_pattern, std_returns_per_pattern, factor_success_rate_for_current_pattern
    
    #path_factors = os.path.realpath("/Users/rhys/Desktop/Heard-Cap/Factor_Project/factors_and_scores.xlsx")
    #path_factors = get_path("factors_and_scores.xlsx")
    #path_patterns = os.path.realpath("/Users/rhys/Desktop/Heard-Cap/Factor_Project/Bull_Bear_Patterns_since_2010.xlsx")
    path_patterns = get_path("Bull_Bear_Patterns_since_2010.xlsx")
    
    #forward_return_df = pd.read_excel(path_factors, index_col="Date")
    patterns = pd.read_excel(path_patterns, index_col="Date")
    
    forward_return_df['patterns'] = patterns
    
    mean_returns_per_pattern = {fact : {} for fact in forward_return_df.drop(columns = ['patterns', 'ry_scores', 'yc_scores'])}
    returns_per_pattern = {fact : {} for fact in forward_return_df.drop(columns = ['patterns', 'ry_scores', 'yc_scores'])}
    std_returns_per_pattern = {fact : {} for fact in forward_return_df.drop(columns = ['patterns', 'ry_scores', 'yc_scores'])}
    
    for fact in forward_return_df.drop(columns = ['patterns', 'ry_scores', 'yc_scores']):

        for p in forward_return_df.patterns.unique():
            
            returns_per_pattern[fact][p] = forward_return_df[fact].loc[(forward_return_df.patterns == p)]
            mean_returns_per_pattern[fact][p] = returns_per_pattern[fact][p].mean()
            std_returns_per_pattern[fact][p] = returns_per_pattern[fact][p].std()
        
    mean_returns_per_pattern = pd.DataFrame(mean_returns_per_pattern)
    
    std_returns_per_pattern = pd.DataFrame(std_returns_per_pattern)
    
    factor_success_rate_for_current_pattern = percent_success(forward_return_df, pattern_now, ['patterns'])
    
def print_to_stdout():
    
    print(f"\n\nCurrent Real Yield Score: {current_real_rate}")
    print(f"Current Yield Curve Score: {current_yield_curve}")
    print(f"Current Yield Curve Pattern: {pattern_now}")
    if abs(current_real_rate) > 3:
        print("\n\n!! RATES ARE BEING SHOCKED !!\n")
    print(f"\n\nMean Returns in Similar Rate Periods:\n")
    print(round(combined_scores_means,3))
    print(f"\n\nMean Returns in {pattern_now} Patterns:\n")
    print(round(mean_returns_per_pattern.T[pattern_now],3))
    print("\n")
    print(f"Percentage of Similar Rate Signals Above Market:\n")
    
    for factor in percent_above_market:
        print(f"{factor} : {percent_above_market[factor]}%")
    
    print("\n")
    
    print(f"Percentage of {pattern_now} Signals Above Market:\n")
    
    for factor in factor_success_rate_for_current_pattern:
        
        if factor not in ['ry_scores', 'yc_scores']:
            print(f"{factor} : {factor_success_rate_for_current_pattern[factor]}%")
    
    print("\n")
    
    print("95% Confidence Return Interval for Similar Rate Environments: \n")
    for num, (m, s) in enumerate(zip(combined_scores_means, combined_scores_std)):
        
        print(f"{combined_scores_means.index[num]} : {round(m - 3*s, 2),round(m + 3*s, 2)}")
        
    print(f"\n95% Confidence Return Interval for a {pattern_now} Yield Curve: \n")
    for num, (m, s) in enumerate(zip(mean_returns_per_pattern.T[pattern_now], std_returns_per_pattern.T[pattern_now])):
        
        print(f"{mean_returns_per_pattern.T[pattern_now].index[num]} : {round(m - 3*s, 2),round(m + 3*s, 2)}")
    
    
    if num_signals >= 35:
        print(f"\n\nSignificance Level: STRONG ({num_signals} signals)") 
    else:
        print(f"\n\nSignificance Level: WEAK ({num_signals} signals)") 
    
    print("\n\n")
    
     
def save():
    
    fig, axs = plt.subplots(2,1,figsize=(10,8))
    
    for num, ax in enumerate(axs):
        
        if num == 0:
            ax.set_title(f"Mean Return in Rate Environments Similar to {(current_real_rate, current_yield_curve)}", size = 12, fontweight='bold')
            ax.bar(combined_scores_means.index, combined_scores_means, yerr = combined_scores_std, alpha=0.5, capsize=10)
        else:
            ax.set_title(f"Mean Return in {pattern_now} Patterns", size = 12, fontweight='bold')
            ax.bar(mean_returns_per_pattern.T[pattern_now].index, mean_returns_per_pattern.T[pattern_now], yerr = mean_returns_per_pattern.T[pattern_now], alpha=0.5, capsize=10)
    
    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    plt.tight_layout()
    plt.savefig(f"{desktop}/exposure_to_rates/rates_and_pattern_returns.png")
    plt.show()

if __name__ == "__main__":
    
    current()
    current_pattern()
    get_pattern_means()
    file_path = get_path("Results.txt")
    file = open(file_path,"w")
    write_to_file(file)
    #print_to_stdout()
    save()
    