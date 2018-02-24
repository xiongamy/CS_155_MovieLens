import numpy as np
import matplotlib.pyplot as plt

def grad_U(Ui, Yij, Vj, Ai, Bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), the floats Ai and Bj (the ith and jth
    elements of bias vectors A and B), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return Ui * eta * reg + Vj * eta * (np.dot(Ui, Vj) - Yij + Ai + Bj)

def grad_V(Vj, Yij, Ui, Ai, Bj, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), the floats Ai and Bj (the ith and jth
    elements of bias vectors A and B), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return Vj * eta * reg + Ui * eta * (np.dot(Ui, Vj) - Yij + Ai + Bj)
    
def grad_AB(Ui, Yij, Vj, Ai, Bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), the floats Ai and Bj (the ith and jth
    elements of bias vectors A and B), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ai (or with respect to Bj; they are equivalent) multiplied by eta.
    """
    return eta * (np.dot(Ui, Vj) - Yij + Ai + Bj)

def get_err(U, V, A, B, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V (with bias vectors A and B)

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    V_norm = np.linalg.norm(V)
    U_norm = np.linalg.norm(U)
    
    reg_err = reg / 2. * (V_norm * V_norm + U_norm * U_norm)
    
    err = 0.
    for y in Y:
        i = y[0] - 1
        j = y[1] - 1
        Yij = y[2]
        diff = Yij - np.dot(U[i], V[j]) - A[i] - B[j]
        err += 0.5 * diff * diff
    
    return (reg_err + err) / len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    # initialize U, V, and bias vectors A, B
    U = np.random.uniform(-0.5, 0.5, (M, K))
    V = np.random.uniform(-0.5, 0.5, (N, K))
    A = np.random.uniform(-0.5, 0.5, M)
    B = np.random.uniform(-0.5, 0.5, N)
    num_y = len(Y)
    
    # train model using stochastic gradient descent
    delta01 = float('inf')
    err = get_err(U, V, A, B, Y, reg=reg)
    loop_through = np.copy(Y)
    for epoch in range(max_epochs):
        np.random.shuffle(loop_through)
        for y in loop_through:
            # update U, V, A, and B
            i = y[0] - 1
            j = y[1] - 1
            Yij = y[2]
            U[i] -= grad_U(U[i], Yij, V[j], A[i], B[j], reg, eta)
            V[j] -= grad_V(V[j], Yij, U[i], A[i], B[j], reg, eta)
            grad_ab = grad_AB(U[i], Yij, V[j], A[i], B[j], reg, eta)
            A[i] -= grad_ab
            B[j] -= grad_ab
            
        # check early stopping condition: loss reduction
        new_err = get_err(U, V, A, B, Y, reg=reg)
        if epoch == 0:
            delta01 = err - new_err
        else:
            delta = err - new_err
            if delta <= delta01 * eps:
                break
        err = new_err
    
    return U, V, A, B, err

		
def main():
    Y_train = np.loadtxt('../data/train.txt').astype(int)
    Y_test = np.loadtxt('../data/test.txt')	.astype(int)
	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20
	
    eta = 0.03 # learning rate
    reg = 1.0
    U, V, A, B, err = train_model(M, N, K, eta, reg, Y_train)
    
    E_in = err
    E_out = get_err(U, V, A, B, Y_test)
    print('E_in:', E_in)
    print('E_out:', E_out)
    
    # save first two latent factors of V
    R, Sigma, S = np.linalg.svd(np.transpose(V))
    tildeV = np.matmul(V, R[:, :2])
    print(tildeV.shape)
    np.savetxt('../results/v_with_bias.csv', tildeV, delimiter=',')
  

if __name__ == "__main__":
    main()
