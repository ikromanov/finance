# Illustration of CAPM model using APPL and VOO

import numpy as np

class Equity():
	def __init__

class CAPM():
	def __init__(self, s_prices, i_prices):
		self.s_prices = s_prices
		self.i_prices = i_prices

	def beta(self):
		prices = np.stack(self.s_prices, self.i_prices)
		return np.cov(prices) / np.var(self.s_prices) * np.var(self.i_prices)

def main():
	


if __name__ == '__main__':
	main()