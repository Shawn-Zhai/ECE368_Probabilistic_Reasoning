import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    a0_range = np.linspace(-1, 1, 100)   
    a1_range = np.linspace(-1, 1, 100)   
    A0, A1 = np.meshgrid(a0_range, a1_range)
    gaussian_contours = []
    x_set = np.zeros((100, 2))
    cov = [[beta, 0], [0, beta]]
    
    # Density Gaussian
    for i in range(0, 100):
        # merge x and y coordinates
        for j in range(100):
            x_set[j][0] = A0[i][j]
            x_set[j][1] = A1[i][j]
            
        gaussian_contours.append(util.density_Gaussian([0, 0], cov, x_set))
    
    # plot
    plt.plot([-0.1], [-0.5], marker = 'o', markersize = 5, color = 'r')
    plt.contour(A0, A1, gaussian_contours, colors = 'b')
    plt.title('prior distribution of weights')
    plt.xlabel('a0')
    plt.ylabel('a1')
    #plt.savefig("prior.pdf")
    plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    
    # Calculations
    column_ones = np.ones((len(x),1))
    X = np.hstack((column_ones, x))
    cov_a_inv = [[1/beta, 0], [0, 1/beta]]
    cov_w_inv = 1/sigma2
    
    mu = np.dot(np.linalg.inv(cov_a_inv + cov_w_inv * np.dot(X.T, X)), (cov_w_inv * np.dot(X.T, z)))
    mu = mu.reshape(2, )
    cov = np.linalg.inv(cov_a_inv + cov_w_inv * np.dot(X.T, X))

    a0_range = np.linspace(-1, 1, 100)   
    a1_range = np.linspace(-1, 1, 100)
    A0, A1 = np.meshgrid(a0_range, a1_range)
    gaussian_contours = []
    x_set = np.zeros((100, 2))

    # Density Gaussian
    for i in range(0, 100):
      # merge x and y coordinates
      for j in range(100):
          x_set[j][0] = A0[i][j]
          x_set[j][1] = A1[i][j]
          
      gaussian_contours.append(util.density_Gaussian(mu.T, cov, x_set))
    
    # plot
    plt.xlabel('a0')
    plt.ylabel('a1')
    
    plt.contour(A0, A1, gaussian_contours, colors = 'b')
    plt.plot([-0.1], [-0.5], marker='o', markersize = 5, color='r')
        
    plt.title(f"posterior distribution of weights with {len(x)} samples")    
    #plt.savefig(f"prediction{len(x_train)}.pdf")
    plt.show()

    return (mu, cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    # Calculations
    column_ones = np.ones((len(x),1))
    X = np.expand_dims(x, 1)
    X = np.hstack((column_ones, X))
    mu_z = np.dot(X, mu)
    
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    
    plt.xlabel('Input')
    plt.ylabel('Prediction')
    
    # Requirements 1,2,3
    std_z = np.sqrt(np.diag(sigma2 + np.dot(np.dot(X, Cov), X.T))) # uncertainty of new inut added
    plt.errorbar(x, mu_z, yerr=std_z, fmt='go')
    plt.scatter(x_train, z_train, color = 'r')
    
    plt.title(f"prediction with {len(x_train)} samples")    
    #plt.savefig(f"prediction{len(x_train)}.pdf")
    plt.show()
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
