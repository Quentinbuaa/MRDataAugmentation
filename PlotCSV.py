import pandas as pd
import matplotlib.pyplot as plt

def read(csv_file):
    data = pd.read_csv(csv_file)
    data.head()
    return data

def get_mrs_data(data, mr_index):
    data = data[data['MR']==mr_index ]
    return data

def plot_acc_mts(data):
    name_dict = mr_name_dict()
    mr_index = str(data['MR'].iloc[0] + 1)
    title = name_dict[mr_index]
    data.plot(x='m',y=['ACC','MTS'], title = title)
    plt.show()

def mr_name_dict():
    name_dict = {
        '1':'Inverting',
        '2':'Rotating (10 degrees)',
        '3':'Scaling (by 90%)'
    }
    return name_dict

def main():
    csv_file = 'result-horizontal.csv'
    data = read(csv_file)
    mr_index = 1
    mr_data = get_mrs_data(data,mr_index )
    plot_acc_mts(mr_data)

if __name__ =="__main__":
    main()

