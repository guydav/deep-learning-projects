from scipy.special import logit, expit
import numpy as np
import tqdm

from .digit_mixer import SklearnDigitMixer
from .abc import default_encoder

DEFAULT_H = 0.5
DEFAULT_EPSILON = 1e-10


def log_p_q(x):
    # for x = logit(p) = log(p / (1 - p))
    # return log(p) and log(q) = log(1-p)
    x = np.array(x)
    x_negative = x < 0
    x_nonnegative = np.logical_not(x_negative)

    log_p = np.zeros_like(x)
    log_q = np.zeros_like(x)

    exp_x = np.exp(x)

    if np.any(x_negative):
        lq = -np.log1p(exp_x[x_negative])
        log_p[x_negative] = lq + x[x_negative]
        log_q[x_negative] = lq

    if np.any(x_nonnegative):
        lp = -np.log1p(1 / exp_x[x_nonnegative])
        log_p[x_nonnegative] = lp
        log_q[x_nonnegative] = lp - exp_x[x_nonnegative]

    return log_p, log_q


def logit_sum(x):
    # for x = logit(p), this returns log(sum(p))

    x = np.array(x)
    x = np.sort(x)[::-1]

    log_p, log_q = log_p_q(x)

    lpm1 = log_p[1:]

    if x[0] < 0:
        lp1 = log_p[0]
        return lp1 + np.log1p(np.sum(np.exp(lpm1 - lp1)))

    else:
        lq1 = log_q[0]
        return np.log1p(-1 * np.exp(lq1) + np.sum(np.exp(lpm1)))


def logit_scale(x, l):
    # For x = logit(p) and l = log(s), this function returns logit(sp).
    x = np.array(x)
    l = np.array(l)

    if len(l.shape) == 0:
        n = x.shape[0]

    else:
        n = max(x.shape[0], l.shape[0])

    if x.shape[0] < n:
        x = np.repeat(x, n / x.shape[0])

    if len(l.shape) == 0:
        l = np.repeat(l, n)

    elif l.shape[0] < n:
        l = np.repeat(l, n / l.shape[0])

    ok_1 = np.logical_and(l < np.log(2), np.abs(l) < np.abs(x + l))
    u = -l - np.logical_not(ok_1) * x
    v = -l - ok_1 * x
    ev = np.exp(v)  # ev is either exp(-(x+l)) or exp(-l) ...
    eumo = np.expm1(u)  # ... and eumo is the other choice minus one

    l2 = np.where(np.isnan(eumo), np.maximum(u, v) + np.log1p(np.exp(-np.abs(u - v))), np.log(eumo + ev))
    return -np.where(v > np.log(2 * np.abs(eumo)), v + np.log1p(eumo / ev), l2)


def diff(x):
    return x[1:] - x[:-1]


def salt_proposal(x, h=DEFAULT_H, epsilon=DEFAULT_EPSILON, seed=None):
    x = np.array(x)
    k = x.shape[0]
    h = np.array(h)
    if len(h.shape) == 0 or len(h) == 1:
        h = np.ones_like(x) * h

    if seed is not None:
        np.random.seed(seed)

    i, l = np.random.choice(np.arange(k), 2)
    x_new = np.copy(x)
    x_new[i] = x[i] + np.random.normal(0, h[i])

    # Calculate logp and logq for old and new draws
    log_p_old_new, log_q_old_new = log_p_q((x[i], x_new[i]))
    # Log of the scaling value
    log_scaling_value = log_q_old_new[1] - logit_sum(x[np.arange(k) != i])
    # Logits of the rescaled simplex point
    x_new[np.arange(k) != i] = logit_scale(x[np.arange(k) != i], log_scaling_value)
    # Add detailed balance term
    log_transition_ratio = diff(log_p_old_new) + (k - 1) * diff(log_q_old_new)

    return x_new, log_transition_ratio


def score_params(params, valid_digits, size, generator, encoder, model, metric,
                 encoded_train):
    generated_data = generator(valid_digits, params)(size)
    encoded_data = encoder(model, generated_data)
    return metric(params, encoded_data, encoded_train)


def abc_mcmc_simplex(valid_digits, train, prior_sampler, model, metric,
                     generator=SklearnDigitMixer, encoder=default_encoder,
                     n_iter=100, n_chains=4, use_tqdm=True, debug=False):
    all_chains = []

    encoded_train = encoder(model, train)

    for c in range(n_chains):
        if debug: print(f'Starting chain #{chain + 1}')

        # n_iter - 1 to arrive at a total of n_iter samples per chain
        # In the real world, there would be burn-in, and a whole host of stuff
        if use_tqdm:
            iterator = tqdm.tnrange(n_iter - 1)
        else:
            iterator = range(n_iter - 1)

        params_0 = prior_sampler(c)
        score_0 = score_params(params_0, valid_digits, train.shape[0], generator,
                               encoder, model, metric, encoded_train)
        chain = [(score_0, params_0)]

        for i in iterator:
            current_score, current_params = chain[-1]
            current_logit = logit(current_params)
            move_logit, log_move_ratio = salt_proposal(current_logit)

            if np.any(np.logical_or(np.isnan(move_logit), np.isinf(move_logit))):
                chain.append(chain[-1])
                continue

            move_params = expit(move_logit)
            move_score = score_params(move_params, valid_digits, train.shape[0], generator,
                                      encoder, model, metric, encoded_train)

            # The question of how to assign probability to distance is interesting
            # I'll got for now with e(-score), which means my log probability
            # is just -score, and my log acceptance ratio is current_score - move_score
            # which intuitively seems reasonably, as it's positive (and improves acceptance probability)
            # when the current score is higher (worse) than the move score, and vice versa
            log_acceptance_ratio = log_move_ratio + current_score - move_score
            accept = np.log(np.random.uniform()) < log_acceptance_ratio

            if accept:
                chain.append((move_score, move_params))
            else:
                chain.append(chain[-1])

        all_chains.append(chain)

    all_chain_results = [item for chain in all_chains for item in chain]
    all_chain_results.sort()

    return all_chain_results
