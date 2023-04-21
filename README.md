# SCSA_JAX

Implementation of the 1-D SCSA algorithm using JAX, numpy and Scipy 

The idea of this project is to improve the runtime of the SCSA-algorithm.
SCSA is an algorithm for signal processing of data vectors that rely on 
the intersection quantum theory and signal processing / applied math. 
The idea of the method is to think in the signal as potential to the
Schroedinger Operator by projecting the signal on its diagonal. The 
eigenproblem solution for this operator can provide us the eigenvalues
and the eigenvectors for the constructed operator. We rely on the eigenfucntions
related to the negative eigenvalues for the reconstruction purpose. 
The scsa have been successfully on denoising and as a feature extraction algorithm
for biosignals.

Things to do: 
https://github.com/empter/PYFEAST

- [] benchmark this implementation on some non synthetic data. 
- [] combine the feast sparse eigensolver with our implementation
     to see if it improves (hopefully it will)

If this ideas works properly, we will be able to increase the usability capabilities
to more insteresting applications.
