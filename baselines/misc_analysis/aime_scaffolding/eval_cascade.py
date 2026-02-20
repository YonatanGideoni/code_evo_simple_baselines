import numpy as np


def compare_accs(accs, others, n_bootstrap=100_000):
    accs = np.asarray(accs)
    others = [np.asarray(o) for o in others]

    # Sample accs: shape (n_bootstrap,)
    acc_samples = np.random.choice(accs, size=n_bootstrap)

    # Sample others: shape (n_bootstrap, n_others)
    other_samples = np.column_stack([
        np.random.choice(o, size=n_bootstrap) for o in others
    ])

    # Max over other accs per bootstrap
    other_max = other_samples.max(axis=1)

    # Comparisons
    greater = acc_samples > other_max
    less = acc_samples < other_max
    tied = acc_samples == other_max

    # Count ties per bootstrap
    n_tied = (other_samples == acc_samples[:, None]).sum(axis=1)

    # Successes
    successes = np.zeros(n_bootstrap, dtype=float)
    successes[greater] = 1.0
    successes[tied] = 1.0 / n_tied[tied]

    return float(successes.mean())


def main():
    # here for illustrative purposes, seeing which of the top-20% (10) IID RS scaffolds is likely best
    agent_accs = [
        [20.0, 26.66666667, 30.0, 20.0, 33.33333333, 20.0, 23.33333333, 30.0, 36.66666667, 26.66666667],
        [26.66666667, 26.66666667, 40.0, 30.0, 20.0, 23.33333333, 30.0, 23.33333333, 30.0, 33.33333333],
        [26.66666667, 26.66666667, 30.0, 30.0, 36.66666667, 26.66666667, 26.66666667, 30.0, 30.0, 36.66666667],
        [36.66666667, 40.0, 23.33333333, 36.66666667, 33.33333333, 23.33333333, 26.66666667, 30.0, 23.33333333,
         33.33333333],
        [23.33333333, 26.66666667, 30.0, 30.0, 26.66666667, 23.33333333, 26.66666667, 30.0, 23.33333333, 23.33333333],
        [33.33333333, 36.66666667, 26.66666667, 36.66666667, 30.0, 33.33333333, 36.66666667, 23.33333333, 30.0, 20.0],
        [23.33333333, 30.0, 23.33333333, 26.66666667, 33.33333333, 26.66666667, 30.0, 23.33333333, 23.33333333,
         26.66666667],
        [23.33333333, 23.33333333, 30.0, 30.0, 30.0, 36.66666667, 36.66666667, 30.0, 30.0, 26.66666667],
        [26.66666667, 23.33333333, 30.0, 33.33333333, 23.33333333, 23.33333333, 33.33333333, 23.33333333, 26.66666667,
         33.33333333],
        [36.66666667, 36.66666667, 26.66666667, 30.0, 23.33333333, 23.33333333, 33.33333333, 26.66666667, 26.66666667,
         36.66666667],
    ]

    prob_dominates = []
    for i in range(len(agent_accs)):
        others = agent_accs[:i] + agent_accs[i + 1:]
        prob = compare_accs(agent_accs[i], others)
        prob_dominates.append(prob)

    best_agent_index = prob_dominates.index(max(prob_dominates))
    best_agent_acc = agent_accs[best_agent_index]
    print(f'Probabilities of dominance: {prob_dominates}')
    print(f'Best scaffold idx: {best_agent_index}')
    print(f'Best generalised probability of improvement: {max(prob_dominates)}')
    print(f"That scaffold's acc (avg+-std): {np.mean(best_agent_acc)} +- {np.std(best_agent_acc)}")
    print(f"That scaffold's accs: {best_agent_acc}")


if __name__ == '__main__':
    main()
