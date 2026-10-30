






import sys, os, time, argparse
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax, jax.numpy as jnp, numpy as np, optax, equinox as eqx
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from model_dwno_direct3d import DWNODirect3d, count_params



ntest = 100
batch_size = 25
learning_rate = 0.001
epochs = 2000
weight_decay = 1e-4

width, level = 64, 4
nx, ny, nz = 64, 64, 48
num_param_levels = 1
num_wavelet_layers = 6

T_amb, delta_T, f_scale = 298.15, 100.0, 0.03

data_dir = os.path.join(_this_dir, '..', '..', 'data', '3d')
base_save_dir = os.path.join(_this_dir, 'results_dwno_direct3d')



def rel_l2_norm(pred, true):
    return jnp.linalg.norm((pred - true).ravel()) / jnp.linalg.norm(true.ravel())


def train(ntrain, k, L):
    device = 'GPU' if jax.default_backend() == 'gpu' else jax.default_backend().upper()
    print(f'JAX backend: {device}')

    suffix = f'k{k}_L{L}' if k is not None else f'L{L}'
    save_dir = os.path.join(base_save_dir, f'H{ntrain}_{suffix}')
    os.makedirs(save_dir, exist_ok=True)

    print('Loading data...')
    s_all = jnp.array(np.load(os.path.join(data_dir, 'f_train.npy')), dtype=jnp.float64)
    u_all = jnp.array(np.load(os.path.join(data_dir, 'u_train.npy')), dtype=jnp.float64)
    s_test = jnp.array(np.load(os.path.join(data_dir, 'f_test.npy')), dtype=jnp.float64)
    u_test = jnp.array(np.load(os.path.join(data_dir, 'u_test.npy')), dtype=jnp.float64)

    s_train, u_train = s_all[:ntrain], u_all[:ntrain]
    s_train, s_test = s_train / f_scale, s_test / f_scale
    u_train, u_test = (u_train - T_amb) / delta_T, (u_test - T_amb) / delta_T

    print(f'Train: {ntrain}, Test: {ntest}, Grid: {nx}x{ny}x{nz}, Batch: {batch_size}')
    print(f'k={k}, L={L}')

    print('Creating model...')
    key = jax.random.PRNGKey(0)
    model = DWNODirect3d(nx=nx, ny=ny, nz=nz, width=width, level=level,
                         num_param_levels=k, num_wavelet_layers=L, key=key)
    n_params = count_params(model)
    print(f'Parameters: {n_params:,}')

    schedule = optax.cosine_decay_schedule(learning_rate, epochs, 0.0)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, src, tgt):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: jnp.mean((m(src) - tgt) ** 2))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    @eqx.filter_jit
    def eval_batch(model, src, tgt):
        return model(src)

    def eval_all(model, src_all, tgt_all, eb=5):
        preds = []
        for start in range(0, src_all.shape[0], eb):
            end = min(start + eb, src_all.shape[0])
            preds.append(np.array(eval_batch(model, src_all[start:end], tgt_all[start:end])))
        pred_all = jnp.concatenate(preds, axis=0)
        pr = pred_all * delta_T + T_amb
        tr = tgt_all * delta_T + T_amb
        d = pr - tr
        return jnp.mean(d**2), rel_l2_norm(pr, tr), pr, jnp.mean(jnp.abs(d)), jnp.max(jnp.abs(d))

    train_loss_log = np.zeros(epochs, dtype=np.float64)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float64)
    test_mae_log = np.zeros(epochs, dtype=np.float64)
    test_max_log = np.zeros(epochs, dtype=np.float64)
    total_time = 0.0
    log_every = max(1, epochs // 100)

    print(f'\n{"="*60}')
    print(f'Training ({epochs} epochs)...')
    print(f'{"="*60}\n')

    for ep in range(epochs):
        t1 = time.perf_counter()
        idx = np.random.choice(ntrain, batch_size, replace=False)
        model, opt_state, loss_val = train_step(model, opt_state, s_train[idx], u_train[idx])

        if ep % log_every == 0 or ep == epochs - 1:
            _, test_rel, _, test_mae, test_max_abs = eval_all(model, s_test, u_test)
            trv, tmv, txv = test_rel.item(), test_mae.item(), test_max_abs.item()
        else:
            trv = test_rel_l2_log[max(0, ep - 1)] if ep > 0 else 1.0
            tmv = test_mae_log[max(0, ep - 1)] if ep > 0 else 0.0
            txv = test_max_log[max(0, ep - 1)] if ep > 0 else 0.0

        t2 = time.perf_counter()
        total_time += (t2 - t1)
        train_loss_log[ep] = loss_val.item()
        test_rel_l2_log[ep] = trv; test_mae_log[ep] = tmv; test_max_log[ep] = txv

        if ep % log_every == 0:
            print(f'Iter {ep:5d} | Time {t2-t1:.2f}s | '
                  f'Train MSE {loss_val.item():.6f} | '
                  f'Test Rel.L2 {trv:.6f} MAE {tmv:.4f}K MaxAbs {txv:.4f}K')

    print(f'\n{"="*60}')
    print(f'Training complete. Total time: {total_time:.1f}s')
    print(f'Final Test Rel.L2: {test_rel_l2_log[-1]:.6f} ({100*test_rel_l2_log[-1]:.4f}%)')
    print(f'Final MAE: {test_mae_log[-1]:.4f}K  MaxAbs: {test_max_log[-1]:.4f}K')
    print(f'{"="*60}\n')

    print('Computing per-sample test error...')
    _, _, pred_all, test_mae, test_max_abs = eval_all(model, s_test, u_test)
    pred_np = np.array(pred_all)
    u_test_raw = np.array(np.load(os.path.join(data_dir, 'u_test.npy')))
    sample_errors = np.zeros(ntest, dtype=np.float64)
    for i in range(ntest):
        sample_errors[i] = rel_l2_norm(jnp.array(pred_np[i]), jnp.array(u_test_raw[i])).item()
    print(f'Mean Testing Error: {100*sample_errors.mean():.4f}%')
    print(f'MAE: {test_mae.item():.4f}K  MaxAbs: {test_max_abs.item():.4f}K')
    print(f'Total Time: {total_time:.1f}s  Params: {n_params:,}')


    np.savez(os.path.join(save_dir, 'train_loss.npz'), train_loss=train_loss_log)
    np.savez(os.path.join(save_dir, 'test_metrics.npz'),
             test_rel_l2=test_rel_l2_log, test_mae=test_mae_log, test_max=test_max_log)
    os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(save_dir, 'model', 'dwno_direct3d.eqx'), model)
    print(f'Results saved to: {save_dir}/')


    z_slices = [0, nz//4, nz//2, 3*nz//4, nz-1]
    n_samples = 4

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(np.arange(epochs), train_loss_log, color='tab:blue', linewidth=2, label='Train MSE')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Train MSE'); ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(epochs), test_rel_l2_log, color='tab:red', linestyle='--', linewidth=2, label='Test Rel.L2')
    ax2.set_ylabel('Test Rel.L2'); ax2.legend(loc='upper right')
    plt.title(f'D_WNO_3D (N={ntrain}, k={k}, L={L})', fontsize=14)
    plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_convergence.png'), dpi=600); plt.close()

    fig, axes = plt.subplots(len(z_slices), n_samples, figsize=(12, 14))
    for row, kz in enumerate(z_slices):
        vmin = min(u_test_raw[:, :, :, kz].min(), pred_np[:, :, :, kz].min())
        vmax = max(u_test_raw[:, :, :, kz].max(), pred_np[:, :, :, kz].max())
        for col in range(n_samples):
            idx = col * (ntest // max(1, n_samples))
            err = 100 * sample_errors[idx]
            axes[row, col].imshow(pred_np[idx, :, :, kz], cmap='gist_ncar', vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f'Pred #{idx} z={kz} err={err:.2f}%', fontsize=9)
            axes[row, col].axis('off')
    plt.suptitle(f'D_WNO_3D Predictions (N={ntrain}, k={k}, L={L})', fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'prediction_slices.png'), dpi=600); plt.close()

    print(f'Plots saved to: {save_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntrain', type=int, default=500)
    ap.add_argument('--num_param_levels', type=int, default=1)
    ap.add_argument('--num_wavelet_layers', type=int, default=10)
    args = ap.parse_args()
    train(args.ntrain, args.num_param_levels, args.num_wavelet_layers)
