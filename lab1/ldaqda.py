import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    # MLE Calculation
    male_count = 0
    female_count = 0
    total_male_height = 0
    total_male_weight = 0
    total_female_height = 0
    total_female_weight = 0
    
    # Data points plotting
    male_height = []
    male_weight = []
    female_height = []
    female_weight = []

    # Calculate male mean, and female mean
    for person in range(len(y)):
        #male
        if y[person] == 1:
            total_male_height += x[person][0]
            total_male_weight += x[person][1]
            male_height.append(x[person][0])
            male_weight.append(x[person][1])
            male_count += 1
        #female
        else:
            total_female_height += x[person][0]
            total_female_weight += x[person][1]
            female_height.append(x[person][0])
            female_weight.append(x[person][1])
            female_count += 1
    
    mu_male = [total_male_height/male_count, total_male_weight/male_count]
    mu_female = [total_female_height/female_count, total_female_weight/female_count]
    
    # calculate cov matrix for LDA
    # |HH HW|
    # |WH WW|
    cov_0_0 = 0
    cov_0_1_and_1_0 = 0 #[1][0] and [0][1] are the same
    cov_1_1 = 0

    for person in range(len(y)):
        
        # male
        if y[person] == 1:
            cov_0_0 += (x[person][0] - mu_male[0]) ** 2
            cov_0_1_and_1_0 += (x[person][0] - mu_male[0]) * (x[person][1] - mu_male[1])
            cov_1_1 += (x[person][1] - mu_male[1]) ** 2
        
        # female
        else:
            cov_0_0 += (x[person][0] - mu_female[0]) ** 2
            cov_0_1_and_1_0 += (x[person][0] - mu_female[0]) * (x[person][1] - mu_female[1])
            cov_1_1 += (x[person][1] - mu_female[1]) ** 2
    
    # normalize by populationsize
    cov = [[cov_0_0/len(y), cov_0_1_and_1_0/len(y)], 
           [cov_0_1_and_1_0/len(y), cov_1_1/len(y)]]
    
    # LDA Plot
    plt.title('LDA Plot')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    
    # Data points
    plt.scatter(male_height, male_weight, color = 'blue')
    plt.scatter(female_height, female_weight, color = 'red')
    
    # Contours
    x_range = np.linspace(50, 80, 100) 
    y_range = np.linspace(80, 280, 100)   
    X, Y = np.meshgrid(x_range, y_range) # shape: (100, )
    
    # Density Gaussian
    lda_m = []
    lda_f = []
    x_set = np.zeros((100, 2))
    for i in range(100):
        # merge x and y coordinates
        for j in range(100):
            x_set[j][0] = X[i][j]
            x_set[j][1] = Y[i][j]
            
        lda_m.append(util.density_Gaussian(mu_male, cov, x_set))
        lda_f.append(util.density_Gaussian(mu_female, cov, x_set))

    #print(np.asarray(male_lda).shape)
    plt.contour(X, Y, np.asarray(lda_m), colors='blue')
    plt.contour(X, Y, np.asarray(lda_f), colors='red')

    # plot the decision boundary
    lda_decision_boundary = np.asarray(lda_m) - np.asarray(lda_f)
    plt.contour(X, Y, lda_decision_boundary, 1, color='black')
    
    plt.show()
    
    # calculate cov matrices for QDA
    cov_male_0_0 = 0
    cov_male_0_1_and_1_0 = 0
    cov_male_1_1 = 0
    cov_female_0_0 = 0
    cov_female_0_1_and_1_0 = 0
    cov_female_1_1 = 0

    #calculate cov matrix for male and female
    for person in range(len(y)):
        # male matrix
        if y[person] == 1:
            cov_male_0_0 += (x[person][0]-mu_male[0]) ** 2
            cov_male_0_1_and_1_0 += (x[person][0]-mu_male[0]) * (x[person][1]-mu_male[1])
            cov_male_1_1 += (x[person][1]-mu_male[1]) ** 2
        
        # female matrix
        else:
            cov_female_0_0 += (x[person][0]-mu_female[0]) ** 2
            cov_female_0_1_and_1_0 += (x[person][0]-mu_female[0]) * (x[person][1]-mu_female[1])
            cov_female_1_1 += (x[person][1]-mu_female[1]) ** 2
        
    # normalize by populationsize
    cov_male = [[cov_male_0_0/male_count, cov_male_0_1_and_1_0/male_count], 
                [cov_male_0_1_and_1_0/male_count, cov_male_1_1/male_count]]
    cov_female = [[cov_female_0_0/female_count, cov_female_0_1_and_1_0/female_count], 
                  [cov_female_0_1_and_1_0/female_count, cov_female_1_1/female_count]]
    
    
    # QDA Plot
    plt.title('QDA Plot')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    
    # Data points
    plt.scatter(male_height, male_weight, color = 'blue')
    plt.scatter(female_height, female_weight, color = 'red')
    
    # Contours
    # Density_Gaussian
    qda_m = []
    qda_f = []
    
    for i in range(100):
        # merge x and y coordinates
        for j in range(100):
            x_set[j][0] = X[i][j]
            x_set[j][1] = Y[i][j]
        qda_m.append(util.density_Gaussian(mu_male, cov_male, x_set))
        qda_f.append(util.density_Gaussian(mu_female, cov_female, x_set))
        
    plt.contour(X, Y, np.asarray(qda_m), colors='blue')
    plt.contour(X, Y, np.asarray(qda_f), colors='red')
    
    #plot the decision boundary
    qda_decision_boundary = np.asarray(qda_m) - np.asarray(qda_f)
    plt.contour(X, Y, qda_decision_boundary, 0, color='black')
    
    plt.show()
    
    # convert to numpy arrays
    return (np.asarray(mu_male),np.asarray(mu_female),np.asarray(cov),np.asarray(cov_male),np.asarray(cov_female))
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
   
    # Calculation of LHS and RHS of LDA decision boundary
    male_lda = np.dot(mu_male.T, np.dot(np.linalg.inv(cov), x.T)) - 1/2*np.dot(mu_male.T, np.dot(np.linalg.inv(cov), mu_male))
    female_lda = np.dot(mu_female.T, np.dot(np.linalg.inv(cov), x.T)) - 1/2*np.dot(mu_female.T, np.dot(np.linalg.inv(cov), mu_female))
    
    # Count mis-classification
    lda_error = 0
    for person in range(0, len(y)):
        
        if (male_lda[person] >= female_lda[person] and y[person] == 2) or (male_lda[person] < female_lda[person] and y[person] == 1):
            lda_error += 1
    
    # Percent Error
    lda_miss_rate = lda_error / len(y)
    
    # Calculation of LHS and RHS of QDA decision boundary
    male_qda = []
    female_qda = []
    for i in range(0, x.shape[0]):
        male_qda.append(-1/2 * np.log(np.linalg.det(cov_male)) - 1/2 * np.dot(x[i], np.dot(np.linalg.inv(cov_male), x[i].T)) + 
                        np.dot(mu_male.T, np.dot(np.linalg.inv(cov_male), x[i].T)) - 1/2 * np.dot(mu_male.T, np.dot(np.linalg.inv(cov_male), mu_male)))
        
        female_qda.append(-1/2 * np.log(np.linalg.det(cov_female)) - 1/2 * np.dot(x[i], np.dot(np.linalg.inv(cov_female), x[i].T)) + 
                          np.dot(mu_female.T, np.dot(np.linalg.inv(cov_female), x[i].T)) - 1/2 * np.dot(mu_female.T, np.dot(np.linalg.inv(cov_female), mu_female))) 
        
    male_qda = np.asarray(male_qda)
    female_qda = np.asarray(female_qda)
    
    # Count mis-classification
    qda_error = 0   
    for person in range(0, len(y)):
        
        if (male_qda[person] >= female_qda[person] and y[person] == 2) or (male_qda[person] <= female_qda[person] and y[person] == 1):
            qda_error += 1
            
    # Percent Error
    qda_miss_rate = qda_error / len(y)
    
    return (lda_miss_rate, qda_miss_rate)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    LDA, QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    print("LDA Misclassification Error:", LDA)
    print("QDA Misclassification Error:", QDA)
    

    
    
    

    
