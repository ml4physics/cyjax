import logging

import flax.serialization
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import cyjax
import time
import optax
import sympy
from functools import partial
from itertools import product as _cart
from omegaconf import DictConfig

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it):
        return it


def cube_param_sampler(key, batch_size, *, max_abs=1, params_count):
    vals = cyjax.random.uniform_components(key, (batch_size, params_count))
    return vals * jnp.array(max_abs)


def plot_accuracies(accuracies, ranges, params, fig=None, vmin=None, vmax=None, marg2d=np.mean, cmap='Spectral_r'):
    if fig is None:
        fig = plt.figure(figsize=(11, 9))
    if vmin is None:
        vmin = np.min(accuracies)
    if vmax is None:
        vmax = np.max(accuracies)

    count = accuracies.ndim
    axs = []

    for i, ((r, p), part) in enumerate(_cart(zip(ranges, params), range(2))):
        if i % 2 == 0:
            ax = plt.subplot2grid((count, count), (i, i))
            axs.append(ax)
        else:
            plt.subplot2grid((count, count), (i, i), sharex=axs[-1])

        axes = tuple(a for a in range(count) if a != i)
        xvals = np.linspace(-r, r, accuracies.shape[i])
        smean = np.mean(accuracies, axis=axes)
        smin = np.min(accuracies, axis=axes)
        smax = np.max(accuracies, axis=axes)

        plt.fill_between(xvals, smin, smax, color='C0', alpha=0.4)
        plt.plot(xvals, smean, '.--', color='C0')
        plt.ylim(vmin, vmax)

        if i == count - 1:
            comp = 'Re' if part == 0 else 'Im'
            plt.xlabel(fr'$\mathrm{{{comp}}}[{sympy.printing.latex(p)}]$')

        if i == 0:
            plt.ylabel(r'$\sigma$ accuracy')

    img = None
    for i, ((r1, p1), part1) in enumerate(_cart(zip(ranges, params), range(2))):
        for j, ((r2, p2), part2) in enumerate(_cart(zip(ranges, params), range(2))):
            if i >= j:
                continue

            ax = plt.subplot2grid((count, count), (j, i), sharex=axs[i // 2])
            ax.label_outer()
            axes = tuple(a for a in range(count) if a != i and a != j)

            marg = marg2d(accuracies, axis=axes)
            delta1 = (2 * r1) / marg.shape[0] / 2
            delta2 = (2 * r2) / marg.shape[1] / 2

            img = plt.imshow(
                marg.T,
                vmin=vmin,
                vmax=vmax,
                origin='lower',
                cmap=cmap,
                extent=(-r1 - delta1, r1 + delta1, -r2 - delta2, r2 + delta2),
                aspect='auto')
            if j == 0:
                comp = 'Re' if part1 == 0 else 'Im'
                plt.ylabel(fr'$\mathrm{{{comp}}}[{sympy.printing.latex(p1)}]$')
            if i == count - 1:
                comp = 'Re' if part2 == 0 else 'Im'
                plt.xlabel(fr'$\mathrm{{{comp}}}[{sympy.printing.latex(p2)}]$')

    plt.tight_layout()

    axlegend = plt.subplot2grid((count, count), (0, 1))
    axlegend.set_frame_on(False)
    axlegend.set_xticks([])
    axlegend.set_yticks([])
    plt.fill_between([-10], [-10], [-10], color='C0', alpha=0.4, label='min/max range')
    plt.plot([-10], [-10], '.--', color='C0', label='mean')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend(loc='center right')

    fig.subplots_adjust(left=0.15)
    cbar_ax = fig.add_axes([.05, .1, 0.01, .85], )
    fig.colorbar(img, cax=cbar_ax, orientation='vertical')
    cbar_ax.yaxis.set_ticks_position('left')
    return fig


@hydra.main(version_base='1.2', config_path="conf", config_name="quintic")
def main(cfg: DictConfig) -> None:
    import os
    print(os.getcwd())
    variety: cyjax.VarietySingle = hydra.utils.instantiate(cfg.variety)
    log.info(f'Training for variety:\n{variety}')

    cfg.algebraic_metric.sections.dim_proj = variety.dim_projective
    metric = hydra.utils.instantiate(cfg.algebraic_metric)
    cfg.network.basis_size = metric.sections.size
    model = hydra.utils.instantiate(cfg.network)
    log.info(f'Using neural network:\n{model}')

    # Define a loss function
    def loss_function(params, key, sample):
        h_params = model.apply(params, sample[0], rngs={'dropout': key})
        hs = jax.vmap(cyjax.ml.cholesky_from_param)(h_params)
        # vmap over different values of psi
        loss = jax.vmap(
            cyjax.ml.variance_eta_loss,
            (0, 0, None)
        )(hs, sample, metric)
        return jnp.mean(loss)

    # Update step
    opt = hydra.utils.instantiate(cfg.training.opt)

    @jax.jit
    def update_step(key, params, opt_state, sample):
        loss, grads = jax.value_and_grad(loss_function)(params, key, sample)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # Initialize pseudo-randomness
    seed = cfg.random_seed
    if seed is None:
        seed = jax.random.PRNGKey(time.time_ns())
    else:
        seed = jax.random.PRNGKey(seed)
    log.info(f'Using random key {seed}')
    rns = cyjax.util.PRNGSequence(seed)

    # Define a sampler
    param_sampler = partial(
        cube_param_sampler,
        params_count=variety.par_count,
        max_abs=cfg.sampling.param_bounds)
    log.info('Initializing sampler...')
    batch_sampler = cyjax.ml.BatchSampler(
        variety=variety,
        params_sampler=param_sampler,
        **cfg.sampling.sizes,
        seed=next(rns))
    log.info('Finished initializing sampler.')

    # For monitoring the accuracy
    moduli_validation = param_sampler(next(rns), cfg.evaluation.validation_count)
    np.save('validation_params', moduli_validation)

    @partial(jax.jit, backend='cpu')
    def validation_accuracy(key, h):
        return metric.sigma_accuracy(key, moduli_validation, h, cfg.evaluation.sample_size)

    @jax.jit
    def get_h_matrix(params):
        h_params = model.apply(params, moduli_validation, deterministic=True)
        return jax.vmap(cyjax.ml.cholesky_from_param)(h_params)

    # Initialize training
    init = partial(model.init, deterministic=True)
    params = jax.jit(init)(next(rns), param_sampler(next(rns), 1))
    opt_state = opt.init(params)
    losses = []
    validation = []

    time_now = time_start = time.time()
    train_time = cfg.training.train_minutes * 60
    log_every = cfg.training.log_every
    step = 0
    while time_now - time_start < train_time:
        sample = next(batch_sampler)
        loss, params, opt_state = update_step(next(rns), params, opt_state, sample)
        losses.append(loss.item())

        if step % log_every == 0:
            acc = validation_accuracy(next(rns), get_h_matrix(params))
            validation.append(acc)
            log.info(f'Accuracy after {step} steps: {np.mean(acc)}')
            np.save('losses', losses)
            np.save('validation', validation)
            with open(f'final.params', 'wb') as save_file:
                save_file.write(flax.serialization.to_bytes(params))

        time_now = time.time()
        step += 1

    if cfg.evaluation.run:
        log.info('Evaluating accuracy after training.')

        def delta(r):
            return (2*r)/(cfg.evaluation.grid_points-1)

        moduli_grid = np.mgrid[sum((
            (np.s_[-r:r+delta(r)/10:delta(r)],) * 2
            for r in cfg.sampling.param_bounds
        ), start=())]

        moduli = moduli_grid.reshape(variety.par_count, 2, -1)
        moduli = (moduli[:, 0, :] + 1j * moduli[:, 1, :]).T

        @partial(jax.vmap, in_axes=(0, 0, 0))
        def accuracies(key, h, par):
            return metric.sigma_accuracy(key, par, h, cfg.evaluation.sample_size)

        # integral involves sampling -> need cpu
        @partial(jax.jit, backend='cpu')
        def eval_accuracies(key, h_params, par):
            h = jax.vmap(cyjax.ml.cholesky_from_param)(h_params)
            sigs = accuracies(jax.random.split(key, len(h)), h, par)
            return sigs

        h_params = model.apply(params, moduli, deterministic=True)
        batch_size = cfg.evaluation.batch_size
        batches = np.ceil(len(moduli)/batch_size).astype(int).item()
        accuracy_parts = []
        for i in tqdm(range(batches)):
            h_batch = h_params[i*batch_size:(i+1)*batch_size]
            moduli_batch = moduli[i*batch_size:(i+1)*batch_size]
            acc = eval_accuracies(next(rns), h_batch, moduli_batch)
            accuracy_parts.append(acc)

        accuracies = np.concatenate(accuracy_parts)
        accuracies = accuracies.reshape(moduli_grid[0].shape)
        np.savez('sigma_final', sigmas=accuracies, moduli_grid=moduli_grid)

        plot_accuracies(accuracies, cfg.sampling.param_bounds, variety.parameters, marg2d=np.min)
        plt.savefig('final-min.pdf')
        plt.close()

        plot_accuracies(accuracies, cfg.sampling.param_bounds, variety.parameters, marg2d=np.max)
        plt.savefig('final-max.pdf')
        plt.close()

        plot_accuracies(accuracies, cfg.sampling.param_bounds, variety.parameters, marg2d=np.mean)
        plt.savefig('final-mean.pdf')
        plt.close()


if __name__ == "__main__":
    main()
