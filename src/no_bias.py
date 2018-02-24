import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return Ui * eta * reg + Vj * eta * (np.dot(Ui, Vj) - Yij)

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return Vj * eta * reg + Ui * eta * (np.dot(Ui, Vj) - Yij)

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

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
        diff = Yij - np.dot(U[i], V[j])
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
    U = np.random.uniform(-0.5, 0.5, (M, K))
    V = np.random.uniform(-0.5, 0.5, (N, K))
    num_y = len(Y)
    
    delta01 = float('inf')
    err = get_err(U, V, Y, reg=reg)
    loop_through = np.copy(Y)
    for epoch in range(max_epochs):
        np.random.shuffle(loop_through)
        for y in loop_through:
            i = y[0] - 1
            j = y[1] - 1
            Yij = y[2]
            U[i] -= grad_U(U[i], Yij, V[j], reg, eta)
            V[j] -= grad_V(V[j], Yij, U[i], reg, eta)
        new_err = get_err(U, V, Y, reg=reg)
        if epoch == 0:
            delta01 = err - new_err
        else:
            delta = err - new_err
            if delta <= delta01 * eps:
                break
        err = new_err
    
    return U, V, err

import matplotlib
matplotlib.use('Agg')
    
import matplotlib.pyplot as plt
		
def main():
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt')	.astype(int)
	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20
	
    regs = [-3, -2, -1, 0, 1, 2, 3]
    eta = 0.03 # learning rate
    E_in = []
    E_out = []
	
    # Use to compute Ein and Eout
    for r in regs:
        reg = 10. ** r
        U,V, err = train_model(M, N, K, eta, reg, Y_train)
        E_in.append(err)
        E_out.append(get_err(U, V, Y_test))
	
    
    plt.plot(regs, E_in, label='$E_{in}$')
    plt.plot(regs, E_out, label='$E_{out}$')
    plt.title('Error vs. Reg')
    plt.xlabel('Reg')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('reg_err.png')	

  

if __name__ == "__main__":
    main()
