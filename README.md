# Matrix-Eigenvectors
Compute matrix and find first two eigenvectors. 

The dataset has 10 real attributes, and the last one is simply the class label, which is categorical, and which you will ignore . Assume that attributes are numbered starting from 0. You should use Python and the NumPy scientiﬁc computing package for answering the following questions.
1: Write a function to Compute the sample covariance matrix as inner products between the columns of the centered data matrix . Show that the result from your function matches the one using numpy.cov function. (Note: Numpy Cov function has a parameter bias, which you should set to ‘True’). 
2: Use linalg.eig to ﬁnd the ﬁrst two dominant eigenvectors of the covariance matrix (that you obtained in Question 1), and reduce the data dimensionality from 10 to 2 by computing the projection of data points along these eigenvectors. Now, compute the variance of the datapoints (in reduced dimensionality) using the subroutine that you wrote for Question 1 (Do not print the projected datapoints on stdout, only print the value of the variance. Also, conﬁrm that the variance is equal to the sum of two dominant eigenvalues.)
