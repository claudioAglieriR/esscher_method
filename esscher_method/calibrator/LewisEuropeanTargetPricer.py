from scipy.integrate import quad

from esscher_method.model.Model import *


class LewisEuropeanTargetPricer():
    def __init__(self,
                 model: Model,
                 N: int = 2 ** 10,
                 limit: int = 1000000,
                 T: float = None,
                 K: float = None,
                 target_price: float = None):
        self.model = model
        self.limit = limit
        self.N = N
        self.T = T
        self.K = K
        self.target_price = target_price
        self.asset_bounds = [(self.target_price, np.iinfo(np.int64).max)]

    def price(self, S0: float) -> float:
        # TODO : Add risk-free rate
        log_moneyness = np.log(S0 / self.K)
        chf = lambda u: self.model.chf(chf_input=u, t=self.T, risk_neutral=True)
        integrand = lambda u: np.real(np.exp(u * log_moneyness * 1j) * chf(u - 0.5j)) * 1 / (u ** 2 + 0.25)
        return S0 - 1 / np.pi * np.sqrt(S0 * self.K) * (quad(integrand, 0, self.N, limit=self.limit)[0])

    def price_residual(self, S0: float) -> float:
        return self.price(S0=S0) - self.target_price

    def absolute_price_residual(self, S0: float) -> float:
        return np.abs(self.price_residual(S0=S0))
