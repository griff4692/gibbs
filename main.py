import os

import argparse
import numpy as np
import pandas as pd
import scipy.stats

C = 2


def sample(positive_prob):
    return 1 if np.random.random() <= positive_prob else 0


def assignment_accuracy(z, y):
    possible_y = [1 if a == 'D' else 0 for a in y]
    acc = _accuracy(z, possible_y)
    possible_y = [0 if a == 'D' else 1 for a in y]
    return max(_accuracy(z, possible_y), acc)


def _accuracy(z, y):
    acc = 0.0
    for a, b in zip(z, y):
        if a == b:
            acc += 1
    return acc / float(len(z))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='~/Desktop')

    parser.add_argument('--component_alpha', type=float, default=1.0)
    parser.add_argument('--component_beta', type=float, default=1.0)

    parser.add_argument('--proportion_alpha', type=float, default=10.0)
    parser.add_argument('--proportion_beta', type=float, default=10.0)

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)

    votes_df = pd.read_csv(os.path.join(args.data_dir, 'senate', 'votes.csv'))
    X = votes_df.to_numpy()
    N = X.shape[0]
    x_dim = X.shape[1]
    y = list(map(
        lambda a: a.split('(')[1].split('-')[0],
        open(os.path.join(args.data_dir, 'senate', 'senators.txt')).readlines()))

    proportion_beta_prior = (args.proportion_alpha, args.proportion_beta)
    theta = 0.5  # assume an equal proportion of Dems and Republicans

    components = []
    for _ in range(C):
        components.append(np.random.beta(args.component_alpha, args.component_beta, size=(x_dim,)))

    z = np.random.random_integers(0, high=1, size=(N, ))

    MAX_ITERS = 10
    for iter in range(MAX_ITERS):
        for n_idx in range(N):
            log_likelihoods = [0.0, 0.0]
            for c_idx in range(C):
                prior_assignment = theta if c_idx == 1 else 1.0 - theta
                log_likelihoods[c_idx] += np.log(prior_assignment)
                for vote_outcome, vote_pi in zip(X[n_idx], components[c_idx]):
                    if vote_outcome > -1:
                        prob = 1 - vote_pi if vote_outcome == 0 else vote_pi
                        log_likelihoods[c_idx] += np.log(prob)
            z_prob = np.exp(log_likelihoods[1]) / (np.exp(log_likelihoods[0]) + np.exp(log_likelihoods[1]))
            z_assignment = sample(z_prob)
            z[n_idx] = z_assignment

        pos_assignments = sum(z)
        neg_assignments = N - pos_assignments
        theta = np.random.beta(args.proportion_alpha + pos_assignments, args.proportion_beta + neg_assignments)

        for cidx in range(C):
            component_x = X[np.where(z == cidx)[0]]
            for vote_idx in range(x_dim):
                component_votes = list(filter(lambda x: x >= 0, component_x[:, vote_idx]))
                pos_votes = sum(component_votes)
                neg_votes = len(component_votes) - pos_votes
                component_sample = np.random.beta(args.component_alpha + pos_votes, args.component_beta + neg_votes)
                components[cidx][vote_idx] = component_sample

        log_joint = 0.0
        log_joint += np.log(scipy.stats.beta(args.proportion_alpha, args.proportion_beta).pdf(theta))

        for cidx in range(C):
            for component_bernoulli in components[cidx]:
                log_joint += np.log(
                    scipy.stats.beta(args.component_alpha, args.component_beta).pdf(component_bernoulli))
        for n in range(N):
            x = X[n, :]
            assignment = z[n]
            assignment_components = components[assignment]
            log_joint += np.log(theta if assignment == 1 else 1.0 - theta)

            for vidx in range(x_dim):
                vote = x[vidx]
                vote_pi = assignment_components[vidx]
                if vote > -1:
                    log_joint += np.log(1.0 - vote_pi) if vote == 0 else np.log(vote_pi)

        print(log_joint)
        print(assignment_accuracy(z, y))
