import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
def welcome():
    print("Welcome to price prediction system")
    input("Press any key to continue...")

def csvcheck():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    print(content_list)
    for file in content_list:
        if file.split('.')[-1]=='csv':
            csv_files.append(file)
    if len(csv_files)==0:
        return 0
    else:
        return csv_files
def get_user_choice_csv(csv_files):
    index=1
    for file in csv_files:
        print(index,' ',file)
        index+=1
    return csv_files[int(input('Choose a csv file'))-1]
def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,label='Training Data')
    plt.plot(X_train,regressionObject.predict(X_train),label='Best Fit')
    plt.scatter(X_test,Y_test,label='Test Data')
    plt.scatter(X_test,Y_pred,label='Predicted Test Data')
    plt.xlabel('Flats(BHK)')
    plt.ylabel('Price(in Lacs)')
    plt.title('Flat prices in Faridabad')
    plt.legend()
    plt.show()
def main():
    welcome()
    try:
        csv_files=csvcheck()
        if csv_files==0:
            raise FileNotFoundError('No csv files in the directory')
        csv_file=get_user_choice_csv(csv_files)
        print(csv_file,'is selected')
        print('Reading file..')
        print('Creating Dataset')
        dataset=pd.read_csv(csv_file)
        print('Dataset Created')

        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        test_data_size=float(input("Enter test data size (between 0 and 1)"))
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_data_size)

        print('Model creation in progress')
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print('Model created')

        print('Predicting test data in trained model')
        Y_pred=regressionObject.predict(X_test)
        
        print('X test','...','Y test','...','Y predicted')
        index=0
        while index<len(X_test):
            print(X_test[index],'...',Y_test[index],'...',Y_pred[index])
            index+=1
        input("Press ENTER key to see above result in graphical format")
        graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("Our model is %2.2f%% accurate"%(r2*100))

        print("Enter flat in BHK to predict there prices seperated by commas")
        flats=[float(e) for e in input().split(',')]
        f=[]
        for e in flats:
            f.append([e])
        flat=np.array(f)

        prices=regressionObject.predict(flat)
        data=pd.DataFrame({'Flats in BHK':flats,'Prices in lacks':prices})
        print(data)
        plt.scatter(flat,prices)
        plt.xlabel('Flats(BHK)')
        plt.ylabel('Price(in lacks)')
        plt.show()
    except FileNotFoundError:
        print('No csv files in the directory')
    
if __name__=='__main__':
    main()
    input()
