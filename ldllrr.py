import torch
import torch.nn as nn

class LDL_LRR:
    '''
    Parameters::
        lam: float, default=1e-3, recommendation={1e-6, 1e-5, ..., 1e-1}
            strength of ranking loss.
        beta: float, default=1, recommendation={1e-3, 1e-2, ..., 1e2}
            strength of L2 regularization. 
    --------------------------------------------------------------
    Methods::
        fit(X, D): train the model
        predict(X): predict label distributions
    --------------------------------------------------------------
    Examples::
        >>> X_train, X_test, D_train, D_test = load('SJAFFE')
        >>> model = LDL_LRR()
        >>> model.fit(X_train, D_train)
        >>> D_pred = model.predict(X_test)
    '''
    def __init__(self, lam=1e-3, beta=1, random_state=123):
        self.lam = lam
        self.beta = beta
        self.random_state = random_state
    
    def compute_ranking_loss(self, Dhat, P, W):
        Phat = torch.sigmoid((Dhat.unsqueeze(-1) - Dhat.unsqueeze(1)) * 100)
        l = ((1 - P) * torch.clip(1 - Phat, 1e-9, 1).log() + P * torch.clip(Phat, 1e-9, 1).log()) * W
        return -l.sum()

    def fit(self, X, D):
        torch.manual_seed(self.random_state)
        X, D = torch.FloatTensor(X), torch.FloatTensor(D)
        P = torch.sigmoid(D.unsqueeze(-1) - D.unsqueeze(1))
        P[P > 0.5], P[P < 0.5] = 1, 0
        W = (D.unsqueeze(-1) - D.unsqueeze(1)).square()
        self.model = nn.Sequential(nn.Linear(X.shape[1], D.shape[1]), nn.Softmax(dim=1))
        params = list(self.model.parameters())
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            Dhat = self.model(X) + 1e-9
            klloss = (-D * Dhat.log()).sum()
            rankloss = self.compute_ranking_loss(Dhat, P, W)
            reg = params[0].square().sum() + params[1].square().sum()
            loss = self.lam / (2 * X.shape[0]) * rankloss + klloss + self.beta / 2 * reg
            if loss.requires_grad:
                loss.backward()
            return loss
        optimizer = torch.optim.LBFGS(params, lr=1e-1, max_iter=500, max_eval=None, tolerance_grad=1e-5,
            tolerance_change=1.4901161193847656e-08, history_size=5, line_search_fn='strong_wolfe')
        optimizer.step(closure)
        return self
    
    def predict(self, X):
        X = torch.FloatTensor(X)
        with torch.no_grad():
            Dhat = self.model(X)
            return Dhat.numpy()