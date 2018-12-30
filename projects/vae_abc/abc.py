from .digit_mixer import SklearnDigitMixer
from .vae import device

import numpy as np
import torch
from scipy.spatial.distance import cdist
import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict


default_digits = (0, 1)
train_images = SklearnDigitMixer(default_digits, (0.3, 0.7))(100)


def uniform_two_param_sampler(seed):
    return np.random.dirichlet((1, 1))


def default_encoder(model, data):
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float).to(device)
        mu, logvar = model.encode(data_tensor)

    return np.concatenate((mu.cpu().numpy(), logvar.cpu().numpy()), axis=1)


def metric(params, generated, train):
    return np.mean(cdist(generated, train))


def abc(valid_digits, train, prior_sampler, model, metric,
        generator=SklearnDigitMixer, encoder=default_encoder,
        n_iter=100, use_tqdm=True):
    encoded_train = encoder(model, train)
    results = []

    if use_tqdm:
        iterator = tqdm.tnrange(n_iter)
    else:
        iterator = range(n_iter)

    for i in iterator:
        params = prior_sampler(i)
        generated_data = generator(valid_digits, params)(train.shape[0])
        encoded_data = encoder(model, generated_data)
        distance = metric(params, encoded_data, encoded_train)
        results.append((distance, params))

    results.sort()
    return results


NUM_TESTS = 100


def randomized_test(prob_generator, abc_method, abc_params, num_tests=NUM_TESTS, seed=33):
    np.random.seed(seed)
    test_results = []

    for i in tqdm.tnrange(num_tests):
        probs = prob_generator(i)
        digits = np.random.choice(np.arange(10), len(probs), replace=False)
        images = SklearnDigitMixer(digits, probs)(100)

        results = abc_method(digits, images, use_tqdm=False, **abc_params)
        test_results.append((digits, probs, results[0]))

    return test_results


def analyze_results(test_results, k=5):
    deviations = np.array([np.abs(np.array(t[1]) - np.array(t[2][1]))
                           for t in test_results])
    mean_devs = np.mean(deviations, 1)
    print(np.mean(mean_devs))
    plt.title('Mean Absolute Deviation')
    plt.hist(mean_devs)
    plt.show()

    print('Best:')
    best_idx = np.argpartition(mean_devs, k)
    for i in best_idx[:k]:
        print(test_results[i])

    print('Worst:')
    worst_idx = np.argpartition(mean_devs, -k)
    for i in worst_idx[-k:]:
        print(test_results[i])

    digit_to_results = defaultdict(list)
    for i in range(NUM_TESTS):
        res = test_results[i]
        for d in res[0]:
            digit_to_results[d].append(mean_devs[i])

    print('MAD per digit:')
    for d in range(10):
        print(f'{d}: {np.mean(digit_to_results[d]):.3f}')


def plot_reconstruction(model, im, n_images=8, image_scale=1.5):
    with torch.no_grad():
        tensor_im = torch.tensor(im, dtype=torch.float).to(device)
        recon_im, mu, logvar = model(tensor_im)

    fig = plt.figure(figsize=(n_images * image_scale, 2 * (image_scale + 0.5)))

    nrows = 2
    ncols = n_images

    for i in range(ncols):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(im[i].reshape(28, 28))

        ax_recon = ax = plt.subplot(nrows, ncols, ncols + i + 1)
        ax.imshow(recon_im[i].reshape(28, 28))
