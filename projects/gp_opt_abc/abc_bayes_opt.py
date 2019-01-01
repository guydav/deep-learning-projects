import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import tqdm
import pyro
import pyro.contrib.gp as gp
pyro.enable_validation(True)


class ABCBayesianOptimizer:
    def __init__(self, abc_routine, x_0, dim, x_proposal_sampler,
                 x_constraint=None, ucb_kappa=2, num_proposal_candidates=5,
                 optimize_proposals=True, proposal_from_previous=True,
                 abc_params=None, device=None, verbose=False):

        self.abc_routine = abc_routine
        self.dim = dim
        self.x_proposal_sampler = x_proposal_sampler
        # Set a default constraint which doesn't actually constrain
        if x_constraint is None:
            x_constraint = constraints.real

        self.x_constraint = x_constraint
        self.ucb_kappa = ucb_kappa
        self.num_proposal_candidates = num_proposal_candidates
        self.optimize_proposals = optimize_proposals
        self.proposal_from_previous = proposal_from_previous and optimize_proposals

        if abc_params is None:
            abc_params = {}
        self.abc_params = abc_params

        self.verbose = verbose
        self.total_iter = 0

        if x_0.dim() == 0:
            x_0 = torch.unsqueeze(x_0, dim=-1)

        if verbose: print('Evaluating initial values')
        y_0 = []
        for x in x_0:
            y_0.append(self.abc_routine(x, **self.abc_params).reshape(1, ))

        y_0 = torch.cat(y_0)

        if verbose: print('Creating GP')
        self.gp = gp.models.GPRegression(x_0, y_0, gp.kernels.Matern52(input_dim=dim),
                                         noise=torch.tensor(0.1), jitter=1.0e-3)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        # TOOD: figure out what do I need to do to get this working on the GPU
        #         self.gp.to(self.device)
        self.gp_optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.001)
        gp.util.train(self.gp, self.gp_optimizer)

        best_y, best_y_idx = torch.min(y_0, -1)
        self.best_x = x_0[best_y_idx]
        self.best_y = best_y
        if verbose: print(f'After initial fitting, best f({self.best_x}) = {self.best_y}')

    def update_posterior(self, x_new, y_new):
        if y_new < self.best_y:
            if self.verbose: print(
                f'Found new best -- old: f({self.best_x}) = {self.best_y}, new: f({x_new}) = {y_new}')
            self.best_x = x_new
            self.best_y = y_new

        X = torch.cat([self.gp.X, x_new])  # incorporate new evaluation
        y = torch.cat([self.gp.y, y_new])
        self.gp.set_data(X, y)

        # TODO: do the gp parameters change? do I need to recreate the optimizer here?
        gp.util.train(self.gp, self.gp_optimizer)

    def lower_confidence_bound(self, x):
        mu, variance = self.gp(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - self.ucb_kappa * sigma

    def find_a_candidate(self, x_init, lower_bound=0, upper_bound=1):
        # transform x to an unconstrained domain
        unconstrained_x_init = transform_to(self.x_constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x])

        def closure():
            minimizer.zero_grad()
            x = transform_to(self.x_constraint)(unconstrained_x)
            x = x.reshape((1, self.dim))
            y = self.lower_confidence_bound(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(self.x_constraint)(unconstrained_x)
        return x.detach().reshape((1, self.dim))

    def propose_next_x(self):
        candidates = []
        values = []

        if self.proposal_from_previous:
            x_init = self.gp.X[-1:]
        else:
            x_init = self.x_proposal_sampler()

        for i in range(self.num_proposal_candidates):
            if self.optimize_proposals:
                x = self.find_a_candidate(x_init)
            else:
                x = x_init
            y = self.lower_confidence_bound(x)
            candidates.append(x)
            values.append(y)
            x_init = self.x_proposal_sampler()  # .to(self.device)

        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        return candidates[argmin]

    def optimize(self, n, should_plot=False):
        # TODO: implement a plotting option, if relevant, for dim < 3

        for i in tqdm.tnrange(n):
            x_min = self.propose_next_x()

            if torch.any(torch.isnan(x_min)):
                print(
                    'Warning. At iteration {self.total_iter + i + 1}, encountered nan, omitting from posterior update...')
                continue

            y_min = self.abc_routine(x_min, **self.abc_params).reshape((1,))  # evaluate f at new point.
            if self.verbose: print(f'At iteration {i+1}, proposed {x_min} => {y_min}')
            self.update_posterior(x_min, y_min)

        self.total_iter += n

        return self.best_x, self.best_y
