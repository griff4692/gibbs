import os

import argparse
import numpy as np
import pandas as pd
import scipy.stats

RANDOM_SEED = 1928
np.random.seed(RANDOM_SEED)


def accuracy(z, y):
    party_map = {'D': 0, 'R': 1}
    y_r = np.array(list(map(lambda a: party_map[a], y)))
    acc = len(np.where(z == y_r)[0]) / float(len(z))
    if acc < 0.5:  # Give the benefit of the doubt about assignments aligning with proper values
        acc = 1.0 - acc
    return acc


def calculate_log_joint(args, X, z, components, thetas):
    log_joint = 0.0
    for cidx in range(args.C):
        for component_bernoulli in components[cidx]:
            log_joint += np.log(
                scipy.stats.beta(args.component_alpha, args.component_beta).pdf(component_bernoulli))
    for n in range(X.shape[0]):
        x = X[n, :]
        assignment = z[n]
        assignment_components = components[assignment]
        log_joint += np.log(thetas[assignment])
        for vidx in range(x_dim):
            vote = x[vidx]
            vote_pi = assignment_components[vidx]
            if vote > -1:
                vote_ll = np.log(1.0 - vote_pi) if vote == 0 else np.log(vote_pi)
                log_joint += vote_ll
    return log_joint


def calculate_log_posterior_predictive(X, components, thetas):
    """
    :param X: Vote Data
    :param components: Posterior sample of Beta^hat
    :param thetas: Posterior sample of theta^hat
    :return: Log Posterior Predictive
    """
    log_pp = 0.0
    for n in range(X.shape[0]):
        assignment_lprobs = [0.0] * len(components)
        x = X[n, :]
        for assignment in range(len(components)):
            assignment_components = components[assignment]
            assignment_lprobs[assignment] += np.log(thetas[assignment])
            for vidx in range(x_dim):
                vote = x[vidx]
                vote_pi = assignment_components[vidx]
                if vote > -1:
                    vote_ll = np.log(1.0 - vote_pi) if vote == 0 else np.log(vote_pi)
                    assignment_lprobs[assignment] += vote_ll
        assignment_probs = [np.exp(p) for p in assignment_lprobs]
        log_pp += np.log(sum(assignment_probs))
    return log_pp


def changed_assignments(a, b):
    return len(np.where(a != b)[0])


def resample_assignments(args, X, components, z, argmax=False):
    for n_idx in range(X.shape[0]):
        log_likelihoods = [0.0] * args.C
        for c_idx in range(args.C):
            log_likelihoods[c_idx] += np.log(thetas[c_idx])
            for vote_outcome, vote_pi in zip(X[n_idx], components[c_idx]):
                if vote_outcome > -1:
                    prob = 1 - vote_pi if vote_outcome == 0 else vote_pi
                    log_likelihoods[c_idx] += np.log(prob)
        likelihoods = [np.exp(l) for l in log_likelihoods]
        lsum = float(sum(likelihoods))
        likelihoods_normalized = [l / lsum for l in likelihoods]
        if argmax:
            z_assignment = np.argmax(np.array(likelihoods_normalized))
        else:
            z_assignment = np.random.choice(args.C, 1, p=likelihoods_normalized)[0]
        z[n_idx] = z_assignment


def run_validation_tests(args, dev_X, dev_y, components, thetas):
    dev_z = np.random.random_integers(0, high=args.C - 1, size=(dev_X.shape[0], ))
    resample_assignments(args, dev_X, components, dev_z, argmax=True)
    dev_log_joint = calculate_log_joint(args, dev_X, dev_z, components, thetas) / float(dev_X.shape[0])
    dev_log_posterior_predictive = calculate_log_posterior_predictive(dev_X, components, thetas) / float(dev_X.shape[0])
    print('Validation Log Joint: {}'.format(dev_log_joint))
    print('Approximation to Posterior Predictive: {}'.format(dev_log_posterior_predictive))
    print('Validation Accuracy: {}'.format(accuracy(dev_z, dev_y)))
    dev_republicans = len([d for d in dev_y if d == 'R'])
    dev_democrats = len([d for d in dev_y if d == 'D'])
    print('Validation Splits: Democrats={}, Republicans={}'.format(dev_democrats, dev_republicans))


def train_dev_splits(x, y, train_fract=0.8):
    data_idxs = np.random.permutation(np.arange(len(x)))
    train_split_idx = round(len(x) * train_fract)

    train_x, train_y, dev_x, dev_y = [], [], [], []
    for count, data_idx in enumerate(data_idxs):
        if count < train_split_idx:
            train_x.append(x[data_idx])
            train_y.append(y[data_idx])
        else:
            dev_x.append(x[data_idx])
            dev_y.append(y[data_idx])

    return (np.array(train_x), train_y), (np.array(dev_x), dev_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=2)
    parser.add_argument('--component_alpha', type=float, default=1.0)
    parser.add_argument('--component_beta', type=float, default=1.0)
    parser.add_argument('--data_dir', default='~/Desktop')
    parser.add_argument('--proportion_alpha', type=float, default=1.0)
    parser.add_argument('-use_validation', default=True, action='store_true')
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--lag', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)

    # Load data and political party affiliations (treat as unobserved latent assignment)
    votes_df = pd.read_csv(os.path.join(args.data_dir, 'senate', 'votes.csv'))
    X = votes_df.to_numpy()
    x_dim = X.shape[1]
    y = list(map(
        lambda a: a.split('(')[1].split('-')[0],
        open(os.path.join(args.data_dir, 'senate', 'senators.txt')).readlines()))

    dev_X, dev_y = None, None
    if args.use_validation:
        (X, y), (dev_X, dev_y) = train_dev_splits(X, y)
    N = X.shape[0]

    # Assume uniform proportions prior
    thetas = np.ones([args.C,]) / float(args.C)
    theta_samples = []

    # Each party has its own probability of voting for each bill.  These are Bernoullis with Beta(1, 1) priors
    components, component_samples = [], []
    for _ in range(args.C):
        component_init = np.random.beta(args.component_alpha, args.component_beta, size=(x_dim,))
        components.append(component_init)
        component_samples.append([])

    z = np.random.random_integers(0, high=args.C - 1, size=(N, ))  # Gets over-written so doesn't really matter

    for iter in range(args.burnin + (args.num_samples - 1) * args.lag):
        prev_z = z.copy()
        resample_assignments(args, X, components, z)

        # Resample assignment proportions from dirichlet prior and current assignment proportions
        dirichlet_alphas = [args.proportion_alpha] * args.C
        for z_assignment in z:
            dirichlet_alphas[z_assignment] += 1
        thetas = np.random.dirichlet(dirichlet_alphas)

        # Update per-vote Bernoulli means separately for each cluster based on current cluster assignments
        for cidx in range(args.C):
            component_x = X[np.where(z == cidx)[0]]
            for vote_idx in range(x_dim):
                component_votes = list(filter(lambda x: x >= 0, component_x[:, vote_idx]))
                pos_votes = sum(component_votes)
                neg_votes = len(component_votes) - pos_votes
                component_sample = np.random.beta(args.component_alpha + pos_votes, args.component_beta + neg_votes)
                components[cidx][vote_idx] = component_sample

        log_joint = calculate_log_joint(args, X, z, components, thetas)
        print('Number of changed assignments: {}'.format(changed_assignments(prev_z, z)))
        print('Normalized Log Joint: {}'.format(log_joint / float(X.shape[0])))

        if (iter + 1) >= args.burnin and (iter + 1) % args.lag == 0:
            print('Collecting sample after {} iterations'.format(iter + 1))
            theta_samples.append(thetas)
            for cidx in range(args.C):
                component_samples[cidx].append(components[cidx])

    sample_component_means = [cs.mean(0) for cs in np.array(component_samples)]
    sample_theta_means = np.array(theta_samples).mean(0)
    args.use_validation and run_validation_tests(args, dev_X, dev_y, sample_component_means, sample_theta_means)
