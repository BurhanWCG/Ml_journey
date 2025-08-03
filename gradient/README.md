# Gradient and Gradient Descent: A Walkthrough

## What is a Gradient?

In simple terms, a **gradient** is a vector that points in the direction of the greatest increase of a function. 

- Imagine you are standing on a hill: the gradient tells you which direction is uphill, and how steep it is.  
- In machine learning, we use the gradient of a **loss function** to find out how to change parameters (weights) to reduce error.

---

## Understanding Multivariable Functions and Partial Derivatives

Many functions depend on multiple variables, like \(f(x, y, z)\). To understand how such a function changes, we look at **partial derivatives**:

- A **partial derivative** measures how the function changes with respect to *one* variable while keeping others constant.
- For example, \(\frac{\partial f}{\partial x}\) shows how \(f\) changes if we vary \(x\) only, holding \(y\) and \(z\) fixed.

---

## How the Gradient Combines Partial Derivatives

The **gradient** of a multivariable function is a vector of all its partial derivatives:

\[
\nabla f = \left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right]
\]

This vector points in the direction where the function \(f\) increases the fastest.

---

## What is Gradient Descent?

**Gradient Descent** is an algorithm to find the minimum of a function — often used to minimize the loss function in machine learning.

- It starts from an initial guess of the parameters (weights).
- It calculates the gradient of the loss function at the current weights.
- Then, it **moves the weights in the opposite direction** of the gradient to reduce the loss.
- This process is repeated **iteratively**, updating weights step-by-step.

---

## How Gradient Descent Works Step-by-Step

1. **Initialize weights** with some random values.
2. Compute the **gradient vector** of the loss function at the current weights.
3. Update the weights using the formula:

\[
w_{\text{new}} = w_{\text{old}} - \alpha \times \nabla L(w_{\text{old}})
\]

where:  
- \(w\) are the weights,  
- \(\alpha\) is the **learning rate** (step size),  
- \(\nabla L(w)\) is the gradient of the loss at weights \(w\).

4. Repeat steps 2 and 3 until the loss converges to a minimum or stops improving.

---

## What is Convergence?

**Convergence** means reaching a point where further updates make very little or no improvement in the loss function. At this point, the weights are near the optimal values that minimize the loss.

---

## Summary

- The **gradient** combines all partial derivatives and shows the direction of steepest increase of a function.
- **Partial derivatives** tell how the function changes with respect to each variable individually.
- **Gradient descent** uses the gradient to iteratively update weights **opposite** to the gradient direction, thus minimizing the loss.
- This process continues until the loss reaches a minimum or converges.

---

## Example in this folder

In the code example here, we use a simple quadratic loss function with three variables. The gradient and gradient descent are implemented step-by-step to illustrate the process and show how the weights move closer to the function's minimum.

Feel free to explore the notebook and the code to see gradient descent in action!

---

*Happy Learning!* 🚀
