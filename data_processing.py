import pandas as pd
from urllib.request import urlretrieve
def load_data(Download = True):
    if Download:
        data_path,_ = urlretrieve(url='https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',filename="./car.csv")
        print('Downloaded to car.csv')
    # buying(购买价: vhigh, high, med, low)
    # maint(维护价: vhigh, high, med, low)
    # doors(几个门: 2, 3, 4, 5,more)
    # persons(载人量: 2, 4, more)
    # lug_boot(贮存空间: small, med, big)
    # safety(安全性: low, med, high)
    columns_name = {'buying','maint','doors','persons','lug_boot','safety','condition'}
    data = pd.read_csv('car.csv',names=columns_name)
    return data

def oneHot(data):
    data = pd.get_dummies(data)
    return data

if __name__ == '__main__':
    data = load_data(Download=False)
    for name in data.keys():
        print(name,pd.unique(data[name]))
    data = oneHot(data)


