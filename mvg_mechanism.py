#---------------------------
# mvg_mechanism.py
# Author: Thee Chanyaswad
#
# Version 1.0
#	- Initial implementation.
#---------------------------



import numpy as np


def compute_precision_budget(m, n, gamma, s2, epsilon, delta):
	maxRank = np.min((n,m))
	harmR1 = _get_harmonic_num(maxRank,1)
	harmR12 = _get_harmonic_num(maxRank,0.5)
	alpha = (harmR1 + harmR12) * (gamma ** 2) + 2*harmR1*gamma*s2
	zeta = np.sqrt(2*np.sqrt(-m*n*np.log(delta))-2*np.log(delta) + m*n)
	beta = 2*((m*n)**0.25)*zeta*harmR1*s2
	
	#get total precision budget
	precisionBudget = ((-beta + np.sqrt(beta**2 + 8*alpha*epsilon))**2) / (4*(alpha**2))
	
	return precisionBudget


def generate_mvg_noise_via_affine_tx(rowCov,colCov):
	n = len(colCov)
	m = len(rowCov)
	mu = np.zeros((n,m))
	sampIid = np.random.normal(0,1,size=mu.shape) # sample iid gauss first
	
	uSig,s,vSig = np.linalg.svd(rowCov)
	sSig = np.diag(np.sqrt(s))
	Bsig = np.inner(uSig,sSig.T)
	uPsi,s,vPsi = np.linalg.svd(colCov)
	sPsi = np.diag(np.sqrt(s))
	Bpsi = np.inner(uPsi,sPsi.T)
	
	#affine tx
	sampMvg = np.inner(np.inner(Bpsi,sampIid.T),Bsig).T
	
	return sampMvg


def generate_mvg_noise_via_multivariate_gaussian(rowCov,colCov):
	cov = np.kron(colCov,rowCov)
	muVec = np.zeros(len(cov))
	sampVect = np.random.multivariate_normal(muVec, cov, size=1)
	sampMvg = sampVect.reshape((len(rowCov),len(colCov)),order='F')
	return sampMvg


def _get_harmonic_num(order,power=1.0):
	if order == 1:
		value = 1.0
	else:
		value = 1.0/(order**power) + _get_harmonic_num(order-1,power=power)
	return value