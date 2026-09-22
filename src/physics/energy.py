def calculate_variational_free_energy(beta, log_prob, spin_array, batch_size, n):
        """
        Calculate variational free energy for each sample:

            F_q(x) = E(x) + (1 / beta) * log q(x)
        """

        energy = calculate_Ising_energy(n, batch_size ,spin_array)
        vfe = energy + (1 / beta) * log_prob

        return vfe


def calculate_Ising_energy(n, batch_size, arr):
        """
        Calculate total energy for each spin configuration.
        """

        arr = arr.view(batch_size, n, n)

        horizontal_interactions = arr * arr.roll(
            shifts=1,
            dims=2
        )

        vertical_interactions = arr * arr.roll(
            shifts=1,
            dims=1
        )

        E = -1 * (
            horizontal_interactions.sum(dim=(1, 2))
            + vertical_interactions.sum(dim=(1, 2))
        )

        return E.float().detach()